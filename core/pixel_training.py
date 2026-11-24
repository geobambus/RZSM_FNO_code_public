"""Fit one pixel-scale model (FNO or LSTM) on SMAP event windows.

The two model families share every step except the network itself: the
spatial/temporal split, the event windows and rainfall channel, the loss,
early stopping and the optimizer. ``train_one_model`` receives the network
through ``build_model``.
"""

import copy
import os
import random

import netCDF4 as nc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from core.precipitation_inputs import append_precipitation_channel, load_interval_series
from core.rzsm_pipeline_config import (
    OBSERVATION_EVENT_SEQUENCE_LENGTH,
    build_observation_event_sequences,
    mask_to_ssm_observations,
    read_pixel_context_series,
    split_boundaries,
)
from core.static_input_profiles import augment_static_inputs

DEFAULT_PATIENCE = 30
DEFAULT_NUM_EPOCHS = 500
DEFAULT_MIN_DELTA = 1e-6
DEFAULT_SPATIAL_VAL_WEIGHT = 0.70
SPATIAL_SPLIT_SEED = 42
SPATIAL_TRAIN_FRACTION = 0.8
STATIC_NOISE_STD = 0.05
STATIC_DROPOUT_RATE = 0.05
SSM_VARIABLE = 'SMAP_SSM'
TARGET_VARIABLE = 'in-situ_RZSM'


def read_pixel_training_series(dataset):
    """Return SSM, in-situ RZSM, dates and the target index range of a pixel file."""
    return read_pixel_context_series(dataset, ssm_variable=SSM_VARIABLE, target_variable=TARGET_VARIABLE)


def _spatial_training_order(pixel_count):
    order = list(range(pixel_count))
    random.Random(SPATIAL_SPLIT_SEED).shuffle(order)
    return order[:int(pixel_count * SPATIAL_TRAIN_FRACTION)]


def spatial_training_pixels(pixel_count):
    """Row positions of the pixels used for fitting; the rest are spatial validation."""
    return set(_spatial_training_order(pixel_count))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_feature_scaler(table, feature_cols):
    return pd.DataFrame([{'Feature': col, 'Mean': table[col].mean(), 'Std': table[col].std()}
                         for col in feature_cols], columns=['Feature', 'Mean', 'Std'])


def apply_feature_scaler(table, scaler):
    for _, row in scaler.iterrows():
        col = row['Feature']
        if col not in table.columns:
            continue
        if pd.notna(row['Std']) and row['Std'] != 0:
            table[col] = (table[col] - row['Mean']) / row['Std']
        else:
            table[col] = 0.0


def fit_feature_scaler(table, feature_cols):
    """Static-feature scaler fitted on the spatial-training pixels only."""
    train = table.iloc[_spatial_training_order(len(table))]
    return compute_feature_scaler(train, feature_cols)


def loss_curve_plot(title, train_loss, val_loss_temp, val_loss_spat, path):
    fig = Figure(figsize=(10, 6), dpi=500)
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    ax.plot(train_loss, label='Training Loss', color='#1f77b4')
    ax.plot(val_loss_temp, label='Temporal Val Loss', color='#2ca02c')
    ax.plot(val_loss_spat, label='Spatial Val Loss', color='#ff7f0e')
    ax.set_title(f'Loss Curve ({title})', fontweight='bold', fontsize=18, pad=15)
    ax.set_xlabel('Epoch', fontweight='bold', fontsize=14)
    ax.set_ylabel('Loss', fontweight='bold', fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(prop={'size': 12})
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)


class SimpleLossFunction(nn.Module):
    """Per-pixel Smooth-L1 reconstruction plus bias and correlation penalties.

    ``pixel_ids`` groups the windows of one pixel. Each term is computed over
    all valid targets of a pixel, then pixels are averaged with equal weight,
    so pixels with more windows do not dominate. Without ``pixel_ids`` every
    window is its own group.
    """

    def __init__(self, beta=0.01, lambda_bias=0.25, lambda_corr=0.05):
        super().__init__()
        self.beta = beta
        self.lambda_bias = lambda_bias
        self.lambda_corr = lambda_corr
        self.loss_fn = nn.SmoothL1Loss(beta=beta, reduction='none')


    def forward(self, pred, target, pixel_ids=None):
        if pred.shape != target.shape:
            raise ValueError(f'Estimation/target shape mismatch: {pred.shape} vs {target.shape}')
        if pred.ndim < 2:
            raise ValueError(f'Expected a batch of temporal windows, received shape {pred.shape}')

        batch_size = pred.shape[0]
        pred_flat = pred.reshape(batch_size, -1)
        target_flat = target.reshape(batch_size, -1)

        if pixel_ids is None:
            group_inverse = torch.arange(batch_size, device=pred.device)
            group_count = batch_size
        else:
            pixel_ids = torch.as_tensor(pixel_ids, device=pred.device).reshape(-1)
            if pixel_ids.numel() != batch_size:
                raise ValueError(f'Expected {batch_size} pixel IDs, received {pixel_ids.numel()}')
            _, group_inverse = torch.unique(pixel_ids, sorted=False, return_inverse=True)
            group_count = int(group_inverse.max().item()) + 1

        valid_mask = torch.isfinite(target_flat)
        valid_float = valid_mask.to(dtype=pred.dtype)

        def sum_by_group(window_values):
            grouped = torch.zeros(group_count, dtype=pred.dtype, device=pred.device)
            return grouped.scatter_add(0, group_inverse, window_values)

        valid_counts = sum_by_group(valid_float.sum(dim=1))
        safe_counts = valid_counts.clamp_min(1.0)
        valid_groups = valid_counts > 0

        zeros = torch.zeros_like(pred_flat)
        pred_safe = torch.where(valid_mask, pred_flat, zeros)
        target_safe = torch.where(valid_mask, target_flat, zeros)

        element_loss = self.loss_fn(pred_safe, target_safe)
        reconstruction_loss = sum_by_group((element_loss * valid_float).sum(dim=1)) / safe_counts

        errors = torch.where(valid_mask, pred_flat - target_flat, zeros)
        pixel_bias = sum_by_group(errors.sum(dim=1)) / safe_counts
        bias_loss = F.smooth_l1_loss(pixel_bias, torch.zeros_like(pixel_bias), beta=self.beta, reduction='none')

        pred_mean = sum_by_group((pred_safe * valid_float).sum(dim=1)) / safe_counts
        target_mean = sum_by_group((target_safe * valid_float).sum(dim=1)) / safe_counts
        pred_centered = torch.where(valid_mask, pred_flat - pred_mean[group_inverse].unsqueeze(1), zeros)
        target_centered = torch.where(valid_mask, target_flat - target_mean[group_inverse].unsqueeze(1), zeros)

        covariance = sum_by_group((pred_centered * target_centered).sum(dim=1))
        pred_sum_squares = sum_by_group(pred_centered.square().sum(dim=1))
        target_sum_squares = sum_by_group(target_centered.square().sum(dim=1))
        variance_epsilon = torch.finfo(pred.dtype).eps
        pred_norm = torch.sqrt(pred_sum_squares.clamp_min(variance_epsilon))
        target_norm = torch.sqrt(target_sum_squares.clamp_min(variance_epsilon))
        correlation_defined = ((valid_counts > 1) & (pred_sum_squares > variance_epsilon)
                               & (target_sum_squares > variance_epsilon))
        correlation = torch.where(correlation_defined, covariance / (pred_norm * target_norm),
                                  torch.zeros_like(covariance)).clamp(-1.0, 1.0)
        correlation_loss = torch.where(correlation_defined, (1.0 - correlation).clamp(0.0, 2.0),
                                       torch.zeros_like(correlation))

        group_loss = reconstruction_loss + self.lambda_bias * bias_loss + self.lambda_corr * correlation_loss
        weights = valid_groups.to(dtype=pred.dtype)
        return (group_loss * weights).sum() / weights.sum().clamp_min(1.0)


def mean_group_loss(batch_losses):
    """Count-weighted epoch mean from ``(loss, group_count)`` pairs."""
    total_groups = sum(group_count for _, group_count in batch_losses)
    if total_groups <= 0:
        return float('inf')
    return sum(loss * group_count for loss, group_count in batch_losses) / total_groups


def combined_validation_loss(temp_loss, spat_loss, spatial_weight=DEFAULT_SPATIAL_VAL_WEIGHT):
    return temp_loss * (1.0 - spatial_weight) + spat_loss * spatial_weight


class EarlyStopping:
    def __init__(self, patience, min_delta):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False
        self.best_model_weights = None


    def __call__(self, val_loss, model):
        if not np.isfinite(val_loss):
            self.early_stop = True
            return
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_model_weights = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


class SequenceDataset(Dataset):
    """Event windows of one pixel: dynamic inputs, static features, target and pixel id."""

    def __init__(self, ssm, y, static_props, sequence_length, time_values,
                 target_start=0, target_stop=None, pixel_id=None):
        self.window_size = int(sequence_length)
        self.pixel_id = None if pixel_id is None else int(pixel_id)
        windows, targets, self.target_indices = build_observation_event_sequences(
            ssm, y, time_values=time_values, target_start=target_start,
            target_stop=target_stop, window_size=self.window_size,
        )
        if len(windows) > 0:
            self.X_dyn = torch.tensor(windows, dtype=torch.float32)
            self.X_stat = torch.tensor(np.tile(static_props, (len(windows), 1)), dtype=torch.float32)
            self.y_seq = torch.tensor(targets, dtype=torch.float32)
        else:
            self.X_dyn = torch.empty((0, self.window_size, windows.shape[-1]), dtype=torch.float32)
            self.X_stat = torch.empty((0, len(static_props)), dtype=torch.float32)
            self.y_seq = torch.empty((0, 1), dtype=torch.float32)


    def __len__(self):
        return len(self.y_seq)


    def __getitem__(self, idx):
        sample = (self.X_dyn[idx].contiguous(), self.X_stat[idx].contiguous(), self.y_seq[idx])
        return sample if self.pixel_id is None else (*sample, self.pixel_id)


def _validation_loss(model, loaders, crit, device, label):
    batches = []
    with torch.no_grad():
        for loader in loaders:
            for x_dyn, x_stat, y, pixel in loader:
                x_dyn, x_stat, y, pixel = x_dyn.to(device), x_stat.to(device), y.to(device), pixel.to(device)
                loss = crit(model(x_dyn, x_stat), y, pixel)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Non-finite {label} loss')
                batches.append((loss.item(), int(torch.unique(pixel).numel())))
    return mean_group_loss(batches)


def train_one_model(build_model, table, feature_cols, scaler, rain, ismn_dir, output_dir,
                    gpu_id, seed, params, *, loss_plot=None, on_epoch=None, num_epochs=None):
    """Fit one seed and save the best-validation weights to ``output_dir/model.pt``.

    build_model(params, num_static, input_channels) returns the untrained network.
    rain = {'input_dir': folder of <pixel>.npz interval files, 'scaler': {'mean','std'}}.
    Returns the checkpoint path and the losses at the best epoch.
    """
    torch.set_num_threads(1)
    set_seed(seed)
    num_epochs = DEFAULT_NUM_EPOCHS if num_epochs is None else num_epochs
    if scaler['Feature'].tolist() != list(feature_cols):
        raise RuntimeError('Scaler feature order does not match the input table')
    scaled = table.copy()
    apply_feature_scaler(scaled, scaler)
    train_pixels = spatial_training_pixels(len(scaled))

    train_sets, temporal_sets, spatial_sets = [], [], []
    for position, (_, row) in enumerate(scaled.iterrows()):
        pixel = f"{int(row['lat_idx'])}_{int(row['lon_idx'])}"
        with nc.Dataset(os.path.join(ismn_dir, f'{pixel}.nc')) as ds:
            ssm, rzsm, dates, start, stop = read_pixel_training_series(ds)
        train_stop, val_stop = split_boundaries(dates)
        if not start < train_stop < val_stop < stop:
            raise RuntimeError(f'{pixel}: invalid target/split boundaries '
                               f'[{start}, {stop}) train={train_stop} validation={val_stop}')
        intervals = load_interval_series(os.path.join(rain['input_dir'], f'{pixel}.npz'), dates)
        rzsm = mask_to_ssm_observations(rzsm, ssm)
        static = row[list(feature_cols)].values.astype(float)

        def dataset(lo, hi):
            ds = SequenceDataset(ssm, rzsm, static, OBSERVATION_EVENT_SEQUENCE_LENGTH, dates,
                                 target_start=lo, target_stop=hi, pixel_id=position)
            append_precipitation_channel(ds, ssm, intervals, rain['scaler'], enabled=True)
            return ds

        if position in train_pixels:
            parts = ((train_sets, start, train_stop), (temporal_sets, train_stop, val_stop))
        else:
            parts = ((spatial_sets, start, val_stop),)
        for collection, lo, hi in parts:
            ds = dataset(lo, hi)
            if len(ds):
                collection.append(ds)

    if not train_sets or not temporal_sets or not spatial_sets:
        raise RuntimeError('Missing training, temporal-validation or spatial-validation windows')

    train_loader = DataLoader(ConcatDataset(train_sets), batch_size=params['batch_size'],
                              shuffle=True, num_workers=0, pin_memory=True)
    temporal_loaders = [DataLoader(ds, batch_size=len(ds), shuffle=False) for ds in temporal_sets]
    spatial_loaders = [DataLoader(ds, batch_size=len(ds), shuffle=False) for ds in spatial_sets]

    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    input_channels = train_sets[0].X_dyn.shape[-1]
    model = build_model(params, len(feature_cols), input_channels).to(device)
    crit = SimpleLossFunction().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=params.get('scheduler_factor', 0.5),
        patience=params.get('scheduler_patience', 10), min_lr=params.get('eta_min', 1e-6))
    stopper = EarlyStopping(DEFAULT_PATIENCE, DEFAULT_MIN_DELTA)

    history = {'training_loss': [], 'temporal_loss': [], 'spatial_loss': [], 'combined_loss': []}
    for epoch in range(num_epochs):
        model.train()
        batches = []
        for x_dyn, x_stat, y, pixel in train_loader:
            x_dyn, x_stat, y, pixel = x_dyn.to(device), x_stat.to(device), y.to(device), pixel.to(device)
            opt.zero_grad()
            noisy_static = augment_static_inputs(x_stat, feature_cols, STATIC_DROPOUT_RATE, STATIC_NOISE_STD)
            loss = crit(model(x_dyn, noisy_static), y, pixel)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Seed {seed}: non-finite training loss at epoch {epoch}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            batches.append((loss.item(), int(torch.unique(pixel).numel())))

        model.eval()
        temporal = _validation_loss(model, temporal_loaders, crit, device, 'temporal-validation')
        spatial = _validation_loss(model, spatial_loaders, crit, device, 'spatial-validation')
        combined = combined_validation_loss(temporal, spatial)
        scheduler.step(combined)
        for key, value in zip(history, (mean_group_loss(batches), temporal, spatial, combined)):
            history[key].append(value)
        if on_epoch is not None:
            on_epoch(epoch + 1, combined)
        stopper(combined, model)
        if stopper.early_stop or np.isnan(combined):
            break

    if not stopper.best_model_weights or not np.isfinite(stopper.best_loss):
        raise RuntimeError(f'Seed {seed} produced no finite checkpoint')
    model.load_state_dict(stopper.best_model_weights)
    os.makedirs(output_dir, exist_ok=True)
    checkpoint = os.path.join(output_dir, 'model.pt')
    torch.save({name: value.detach().cpu() for name, value in model.state_dict().items()}, checkpoint)
    best = history['combined_loss'].index(stopper.best_loss)
    pd.DataFrame({'epoch': np.arange(1, len(history['combined_loss']) + 1), **history}).to_csv(
        os.path.join(output_dir, 'history.csv'), index=False)
    if loss_plot is not None:
        loss_curve_plot(f'Seed_{seed}', history['training_loss'], history['temporal_loss'],
                        history['spatial_loss'], loss_plot)
    return {'checkpoint': checkpoint, 'seed': seed, 'combined_loss': float(stopper.best_loss),
            'epoch': best + 1, 'temporal_loss': float(history['temporal_loss'][best]),
            'spatial_loss': float(history['spatial_loss'][best])}
