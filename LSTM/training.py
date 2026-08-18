"""Optuna optimization and deterministic ensemble training for the LSTM."""

import copy
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data import build_training_datasets, compute_scaler
from .model import LSTM
from .settings import (
    MODEL_SEEDS,
    OPTUNA_SEED,
    validate_product,
)


DEFAULT_BATCH_SIZE = 1024
DEFAULT_FINAL_EPOCHS = 500
DEFAULT_OPTUNA_EPOCHS = 50
DEFAULT_PATIENCE = 30
DEFAULT_MIN_DELTA = 1e-6
SPATIAL_VALIDATION_WEIGHT = 0.70
TRAINING_RECIPE_VERSION = "matched_fno_protocol_v1"


def set_seed(seed):
    """Set deterministic Python, NumPy, and PyTorch seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class LSTMLoss(nn.Module):
    """Smooth-L1, bias, and correlation loss averaged across point pixels."""

    def __init__(self, beta=0.01, bias_weight=0.25, correlation_weight=0.05):
        super().__init__()
        self.beta = float(beta)
        self.bias_weight = float(bias_weight)
        self.correlation_weight = float(correlation_weight)
        self.smooth_l1 = nn.SmoothL1Loss(beta=self.beta, reduction="none")

    def forward(self, prediction, target, pixel_ids):
        if prediction.shape != target.shape:
            raise ValueError(
                f"Prediction/target shape mismatch: {prediction.shape} vs "
                f"{target.shape}"
            )
        batch_size = prediction.shape[0]
        prediction = prediction.reshape(batch_size, -1)
        target = target.reshape(batch_size, -1)
        pixel_ids = torch.as_tensor(
            pixel_ids, device=prediction.device
        ).reshape(-1)
        if len(pixel_ids) != batch_size:
            raise ValueError("Pixel ID and batch lengths differ.")
        _, group_inverse = torch.unique(
            pixel_ids, sorted=False, return_inverse=True
        )
        group_count = int(group_inverse.max().item()) + 1

        valid = torch.isfinite(target)
        valid_float = valid.to(dtype=prediction.dtype)
        zeros = torch.zeros_like(prediction)
        prediction_safe = torch.where(valid, prediction, zeros)
        target_safe = torch.where(valid, target, zeros)

        def sum_by_group(window_values):
            grouped = torch.zeros(
                group_count, dtype=prediction.dtype, device=prediction.device
            )
            return grouped.scatter_add(0, group_inverse, window_values)

        valid_counts = sum_by_group(valid_float.sum(dim=1))
        safe_counts = valid_counts.clamp_min(1.0)
        reconstruction = sum_by_group(
            (
                self.smooth_l1(prediction_safe, target_safe) * valid_float
            ).sum(dim=1)
        ) / safe_counts

        error = torch.where(valid, prediction - target, zeros)
        bias = sum_by_group(error.sum(dim=1)) / safe_counts
        bias_loss = F.smooth_l1_loss(
            bias,
            torch.zeros_like(bias),
            beta=self.beta,
            reduction="none",
        )

        prediction_mean = sum_by_group(
            (prediction_safe * valid_float).sum(dim=1)
        ) / safe_counts
        target_mean = sum_by_group(
            (target_safe * valid_float).sum(dim=1)
        ) / safe_counts
        prediction_centered = torch.where(
            valid,
            prediction - prediction_mean[group_inverse].unsqueeze(1),
            zeros,
        )
        target_centered = torch.where(
            valid,
            target - target_mean[group_inverse].unsqueeze(1),
            zeros,
        )
        covariance = sum_by_group(
            (prediction_centered * target_centered).sum(dim=1)
        )
        prediction_squares = sum_by_group(
            prediction_centered.square().sum(dim=1)
        )
        target_squares = sum_by_group(target_centered.square().sum(dim=1))
        epsilon = torch.finfo(prediction.dtype).eps
        correlation_defined = (
            (valid_counts > 1)
            & (prediction_squares > epsilon)
            & (target_squares > epsilon)
        )
        denominator = torch.sqrt(
            prediction_squares.clamp_min(epsilon)
        ) * torch.sqrt(target_squares.clamp_min(epsilon))
        correlation = torch.where(
            correlation_defined,
            covariance / denominator,
            torch.zeros_like(covariance),
        ).clamp(-1.0, 1.0)
        correlation_loss = torch.where(
            correlation_defined,
            (1.0 - correlation).clamp(0.0, 2.0),
            torch.zeros_like(correlation),
        )

        group_loss = (
            reconstruction
            + self.bias_weight * bias_loss
            + self.correlation_weight * correlation_loss
        )
        group_weights = (valid_counts > 0).to(dtype=prediction.dtype)
        return (
            (group_loss * group_weights).sum()
            / group_weights.sum().clamp_min(1.0)
        )


class EarlyStopping:
    """Retain the best weights and stop after a fixed non-improvement count."""

    def __init__(self, patience=DEFAULT_PATIENCE, min_delta=DEFAULT_MIN_DELTA):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.best_weights = None
        self.should_stop = False

    def update(self, validation_loss, model):
        if not np.isfinite(validation_loss):
            self.should_stop = True
            return
        if validation_loss < self.best_loss - self.min_delta:
            self.best_loss = float(validation_loss)
            self.counter = 0
            self.best_weights = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
            self.should_stop = self.counter >= self.patience


def _mean_group_loss(losses):
    """Return a count-weighted mean from ``(loss, pixel_count)`` pairs."""
    if not losses:
        return float("inf")
    group_count = sum(count for _, count in losses)
    if group_count <= 0:
        return float("inf")
    return sum(loss * count for loss, count in losses) / group_count


def _validation_loss(model, datasets, loss_function, device):
    """Evaluate one complete pixel at a time so pixels receive equal weight."""
    losses = []
    with torch.no_grad():
        for dataset in datasets:
            loader = DataLoader(
                dataset,
                batch_size=len(dataset),
                shuffle=False,
                num_workers=0,
                pin_memory=str(device).startswith("cuda"),
            )
            for dynamic, static, target, pixel_ids in loader:
                prediction = model(dynamic.to(device), static.to(device))
                pixel_ids = pixel_ids.to(device)
                loss = loss_function(
                    prediction, target.to(device), pixel_ids
                )
                if not torch.isfinite(loss):
                    return float("inf")
                losses.append(
                    (
                        float(loss.item()),
                        int(torch.unique(pixel_ids).numel()),
                    )
                )
    return _mean_group_loss(losses)


def fit_model(
    product,
    input_csv,
    ismn_root,
    scaler,
    hyperparameters,
    seed,
    device="cpu",
    num_epochs=DEFAULT_FINAL_EPOCHS,
    optuna_trial=None,
):
    """Fit one model using the fixed temporal/spatial validation experiment."""
    product = validate_product(product)
    set_seed(seed)
    datasets = build_training_datasets(input_csv, ismn_root, product, scaler)
    batch_size = int(hyperparameters.get("batch_size", DEFAULT_BATCH_SIZE))
    pin_memory = str(device).startswith("cuda")
    training_loader = DataLoader(
        datasets["training"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=pin_memory,
    )

    model = LSTM(
        hidden_size=int(hyperparameters.get("hidden_size", 64)),
        num_static_properties=len(scaler["feature_names"]),
        num_layers=int(hyperparameters.get("num_layers", 2)),
        dropout_static=float(hyperparameters.get("dropout_static", 0.2)),
        dropout_fc=float(hyperparameters.get("dropout_fc", 0.2)),
    ).to(device)
    loss_function = LSTMLoss().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(hyperparameters["learning_rate"]),
        weight_decay=float(hyperparameters["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(hyperparameters.get("scheduler_factor", 0.5)),
        patience=int(hyperparameters.get("scheduler_patience", 10)),
        min_lr=float(hyperparameters.get("eta_min", 1e-6)),
    )
    early_stopping = EarlyStopping()
    history = []

    for epoch in range(int(num_epochs)):
        model.train()
        training_losses = []
        for dynamic, static, target, pixel_ids in training_loader:
            dynamic = dynamic.to(device)
            static = static.to(device)
            target = target.to(device)
            pixel_ids = pixel_ids.to(device)
            optimizer.zero_grad()

            keep = (torch.rand_like(static) > 0.05).float()
            noisy_static = (
                static * keep + torch.randn_like(static) * 0.05 * keep
            )
            prediction = model(dynamic, noisy_static)
            loss = loss_function(prediction, target, pixel_ids)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"LSTM {product} produced non-finite training loss at "
                    f"epoch {epoch}."
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            training_losses.append(
                (float(loss.item()), int(torch.unique(pixel_ids).numel()))
            )

        model.eval()
        temporal_loss = _validation_loss(
            model,
            datasets["temporal_validation"],
            loss_function,
            device,
        )
        spatial_loss = _validation_loss(
            model,
            datasets["spatial_validation"],
            loss_function,
            device,
        )
        validation_loss = (
            (1.0 - SPATIAL_VALIDATION_WEIGHT) * temporal_loss
            + SPATIAL_VALIDATION_WEIGHT * spatial_loss
        )
        scheduler.step(validation_loss)
        history.append(
            {
                "epoch": epoch + 1,
                "training_loss": _mean_group_loss(training_losses),
                "temporal_validation_loss": temporal_loss,
                "spatial_validation_loss": spatial_loss,
                "combined_validation_loss": validation_loss,
            }
        )

        if optuna_trial is not None:
            optuna_trial.report(validation_loss, epoch)
            if optuna_trial.should_prune():
                import optuna

                raise optuna.TrialPruned()
        early_stopping.update(validation_loss, model)
        if early_stopping.should_stop:
            break

    if early_stopping.best_weights is None:
        raise RuntimeError(f"Training did not produce valid weights for {product}.")
    model.load_state_dict(early_stopping.best_weights)
    state_dict = {
        name: value.detach().cpu() for name, value in model.state_dict().items()
    }
    return state_dict, early_stopping.best_loss, history


def optimize_hyperparameters(
    product,
    input_csv,
    ismn_root,
    *,
    n_trials=200,
    num_epochs=DEFAULT_OPTUNA_EPOCHS,
    device="cpu",
    storage=None,
):
    """Run the active TPE search and return best parameters plus the scaler."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    product = validate_product(product)
    input_frame = pd.read_csv(input_csv)
    scaler = compute_scaler(input_frame)

    def objective(trial):
        hyperparameters = {
            "learning_rate": trial.suggest_float(
                "learning_rate", 5e-5, 2e-3, log=True
            ),
            "weight_decay": trial.suggest_float(
                "weight_decay", 1e-6, 5e-2, log=True
            ),
            "hidden_size": trial.suggest_categorical(
                "hidden_size", [32, 64, 128]
            ),
            "num_layers": trial.suggest_int("num_layers", 1, 3),
            "dropout_static": trial.suggest_float(
                "dropout_static",
                0.15 if product == "ASCAT" else 0.05,
                0.70 if product == "ASCAT" else 0.60,
            ),
            "dropout_fc": trial.suggest_float(
                "dropout_fc",
                0.15 if product == "ASCAT" else 0.05,
                0.70 if product == "ASCAT" else 0.60,
            ),
            "batch_size": DEFAULT_BATCH_SIZE,
            "eta_min": 1e-6,
            "scheduler_factor": 0.5,
            "scheduler_patience": 10,
        }
        _, validation_loss, _ = fit_model(
            product,
            input_csv,
            ismn_root,
            scaler,
            hyperparameters,
            OPTUNA_SEED,
            device=device,
            num_epochs=num_epochs,
            optuna_trial=trial,
        )
        return validation_loss

    study = optuna.create_study(
        study_name=f"LSTM_{product}_{TRAINING_RECIPE_VERSION}",
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=OPTUNA_SEED),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=20, n_warmup_steps=30
        ),
        storage=storage,
        load_if_exists=storage is not None,
    )
    study.optimize(objective, n_trials=int(n_trials))
    best_parameters = {
        **study.best_params,
        "batch_size": DEFAULT_BATCH_SIZE,
        "eta_min": 1e-6,
        "scheduler_factor": 0.5,
        "scheduler_patience": 10,
    }
    return best_parameters, scaler, study


def _fit_seed_task(arguments):
    product, input_csv, ismn_root, scaler, hyperparameters, seed, device, epochs = (
        arguments
    )
    state_dict, best_loss, _ = fit_model(
        product,
        input_csv,
        ismn_root,
        scaler,
        hyperparameters,
        seed,
        device=device,
        num_epochs=epochs,
    )
    return seed, state_dict


def train_ensemble(
    product,
    input_csv,
    ismn_root,
    scaler,
    hyperparameters,
    *,
    seeds=MODEL_SEEDS,
    device="cpu",
    num_epochs=DEFAULT_FINAL_EPOCHS,
    workers=1,
):
    """Train every deterministic seed and return results in seed order."""
    product = validate_product(product)
    seeds = tuple(int(seed) for seed in seeds)
    arguments = [
        (
            product,
            input_csv,
            ismn_root,
            scaler,
            hyperparameters,
            seed,
            device,
            int(num_epochs),
        )
        for seed in seeds
    ]
    if int(workers) == 1:
        results = [_fit_seed_task(argument) for argument in arguments]
    else:
        results = []
        with ProcessPoolExecutor(max_workers=min(int(workers), len(seeds))) as pool:
            futures = [pool.submit(_fit_seed_task, argument) for argument in arguments]
            for future in as_completed(futures):
                results.append(future.result())
    by_seed = {seed: state_dict for seed, state_dict in results}
    return [by_seed[seed] for seed in seeds]


def save_ensemble_bundle(
    output_file,
    ensemble,
    scaler,
    hyperparameters,
):
    """Atomically save the complete prediction-ready ensemble in one file."""
    payload = {
        "hyperparameters": dict(hyperparameters),
        "scaler": scaler,
        "state_dicts": list(ensemble),
        "training_recipe_version": TRAINING_RECIPE_VERSION,
    }
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    temporary_file = f"{output_file}.tmp.{os.getpid()}"
    torch.save(payload, temporary_file)
    os.replace(temporary_file, output_file)
    return output_file


def load_ensemble_bundle(bundle_file):
    """Load and validate one prediction-ready LSTM ensemble bundle."""
    bundle = torch.load(bundle_file, map_location="cpu", weights_only=True)
    required = {
        "hyperparameters",
        "scaler",
        "state_dicts",
        "training_recipe_version",
    }
    missing = required.difference(bundle)
    if missing:
        raise RuntimeError(f"LSTM bundle is missing fields: {sorted(missing)}")
    if bundle["training_recipe_version"] != TRAINING_RECIPE_VERSION:
        raise RuntimeError(
            "LSTM bundle uses training recipe "
            f"{bundle['training_recipe_version']!r}; expected "
            f"{TRAINING_RECIPE_VERSION!r}. Rerun LSTM_train.ipynb before "
            "prediction."
        )
    if not bundle["state_dicts"]:
        raise RuntimeError("LSTM bundle contains no trained models.")
    return bundle
