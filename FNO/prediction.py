"""Apply a bundled FNO ensemble to point-scale SSM records."""

import multiprocessing as mp
import os

import netCDF4 as nc
import numpy as np
import torch

from .model import FNO
from .settings import (
    STATIC_FEATURES,
    TARGET_VARIABLE,
    WINDOW_SIZE,
    build_observation_sequences,
    read_point_series,
    summarize_ensemble,
    validate_product,
)
from .training import load_ensemble_bundle


_WORKER_CONTEXT = None


def _scaled_static_values(row, scaler):
    feature_names = list(scaler["feature_names"])
    if feature_names != list(STATIC_FEATURES):
        raise ValueError("Bundle scaler does not use the ordered FNO features.")
    raw = np.asarray([row[name] for name in feature_names], dtype=float)
    mean = np.asarray(scaler["mean"], dtype=float)
    standard_deviation = np.asarray(scaler["standard_deviation"], dtype=float)
    scaled = np.zeros_like(raw)
    np.divide(
        raw - mean,
        standard_deviation,
        out=scaled,
        where=standard_deviation != 0,
    )
    if not np.all(np.isfinite(scaled)):
        raise ValueError("Scaled static predictors contain non-finite values.")
    return scaled


def _models_from_bundle(bundle):
    hyperparameters = bundle["hyperparameters"]
    models = []
    for state_dict in bundle["state_dicts"]:
        model = FNO(
            modes=int(hyperparameters["modes"]),
            width=int(hyperparameters["width"]),
            num_static_properties=len(bundle["scaler"]["feature_names"]),
            dropout_static=float(hyperparameters.get("dropout_static", 0.0)),
            dropout_fc=float(hyperparameters.get("dropout_fc", 0.0)),
        )
        model.load_state_dict(state_dict)
        model.eval()
        models.append(model)
    return models


def initialize_prediction_worker(bundle_file, product):
    """Load a product ensemble once in each point-prediction worker."""
    global _WORKER_CONTEXT
    product = validate_product(product)
    torch.set_num_threads(1)
    bundle = load_ensemble_bundle(bundle_file)
    _WORKER_CONTEXT = {
        "bundle": bundle,
        "models": _models_from_bundle(bundle),
        "product": product,
    }


def _write_prediction_file(
    output_file,
    source_dataset,
    context_indices,
    product,
    ssm,
    observed_rzsm,
    mean_prediction,
    prediction_standard_deviation,
):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    temporary_file = f"{output_file}.tmp.{os.getpid()}"
    try:
        source_time = source_dataset.variables["time"]
        time_values = np.asarray(source_time[:])[context_indices]
        with nc.Dataset(temporary_file, "w", format="NETCDF4") as output:
            output.createDimension("time", len(time_values))
            time = output.createVariable("time", source_time.datatype, ("time",))
            time[:] = time_values
            for attribute in source_time.ncattrs():
                if attribute != "_FillValue":
                    time.setncattr(attribute, source_time.getncattr(attribute))

            surface = output.createVariable(
                f"{product}_SSM", "f4", ("time",), fill_value=np.nan
            )
            surface[:] = ssm
            surface.units = "m3 m-3"
            surface.long_name = f"{product} surface soil moisture"

            observed = output.createVariable(
                "RZSM", "f4", ("time",), fill_value=np.nan
            )
            observed[:] = observed_rzsm
            observed.units = "m3 m-3"
            observed.long_name = "Observed in-situ root-zone soil moisture"

            prediction = output.createVariable(
                "RZSM_prediction", "f4", ("time",), fill_value=np.nan
            )
            prediction[:] = mean_prediction
            prediction.units = "m3 m-3"
            prediction.long_name = "Mean FNO RZSM prediction across model seeds"

            uncertainty = output.createVariable(
                "RZSM_prediction_std", "f4", ("time",), fill_value=np.nan
            )
            uncertainty[:] = prediction_standard_deviation
            uncertainty.units = "m3 m-3"
            uncertainty.long_name = (
                "Sample standard deviation of FNO predictions across model seeds"
            )
            uncertainty.ddof = 1
        os.replace(temporary_file, output_file)
    finally:
        if os.path.exists(temporary_file):
            os.remove(temporary_file)


def predict_point_task(task):
    """Predict one point and atomically write its compact NetCDF output."""
    if _WORKER_CONTEXT is None:
        raise RuntimeError("Prediction worker was not initialized.")
    product = _WORKER_CONTEXT["product"]
    bundle = _WORKER_CONTEXT["bundle"]
    models = _WORKER_CONTEXT["models"]
    row = task["row"]
    point_file = task["point_file"]
    output_file = task["output_file"]
    pixel = f"{int(row['lat_idx'])}_{int(row['lon_idx'])}"
    if not os.path.exists(point_file):
        return {"pixel": pixel, "status": "failed", "reason": "missing point file"}

    try:
        static_values = _scaled_static_values(row, bundle["scaler"])
        with nc.Dataset(point_file) as point_dataset:
            ssm, observed_rzsm, dates, context_indices = read_point_series(
                point_dataset, product, TARGET_VARIABLE
            )
            dynamic, _, target_indices = build_observation_sequences(
                ssm,
                time_values=dates,
                target_start=0,
                target_stop=len(dates),
                window_size=WINDOW_SIZE,
            )
            if len(dynamic) == 0:
                return {
                    "pixel": pixel,
                    "status": "failed",
                    "reason": f"fewer than {WINDOW_SIZE} valid SSM observations",
                }
            dynamic_tensor = torch.tensor(dynamic, dtype=torch.float32)
            static_tensor = torch.tensor(
                np.tile(static_values, (len(dynamic), 1)), dtype=torch.float32
            )
            ensemble_predictions = []
            with torch.no_grad():
                for model in models:
                    event_predictions = (
                        model(dynamic_tensor, static_tensor).reshape(-1).numpy()
                    )
                    prediction = np.full(len(dates), np.nan, dtype=float)
                    prediction[target_indices] = event_predictions
                    ensemble_predictions.append(prediction)
            mean_prediction, prediction_standard_deviation = summarize_ensemble(
                ensemble_predictions
            )
            _write_prediction_file(
                output_file,
                point_dataset,
                context_indices,
                product,
                ssm,
                observed_rzsm,
                mean_prediction,
                prediction_standard_deviation,
            )
    except Exception as error:
        return {"pixel": pixel, "status": "failed", "reason": str(error)}
    return {"pixel": pixel, "status": "succeeded", "reason": ""}


def run_prediction_tasks(tasks, bundle_file, product, workers=1):
    """Run prepared point tasks and return structured success/failure records."""
    tasks = list(tasks)
    if not tasks:
        return []
    worker_count = min(max(1, int(workers)), len(tasks))
    if worker_count == 1:
        initialize_prediction_worker(bundle_file, product)
        return [predict_point_task(task) for task in tasks]

    context = mp.get_context("spawn")
    with context.Pool(
        processes=worker_count,
        initializer=initialize_prediction_worker,
        initargs=(bundle_file, product),
        maxtasksperchild=100,
    ) as pool:
        return list(pool.imap_unordered(predict_point_task, tasks))
