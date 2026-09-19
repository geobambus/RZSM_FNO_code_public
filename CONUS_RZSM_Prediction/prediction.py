"""Generate compact CONUS RZSM products from EF, FNO, and LSTM models."""

import os
from pathlib import Path

import netCDF4 as nc
import numpy as np
import torch
from tqdm.auto import tqdm

from EF.filter import exponential_filter
from FNO.model import FNO
from FNO.training import load_ensemble_bundle as load_fno_bundle
from LSTM.model import LSTM
from LSTM.settings import (
    CONTEXT_END_DATE,
    CONTEXT_START_DATE,
    STATIC_FEATURES,
    WINDOW_SIZE,
    build_observation_sequences,
    context_period_indices,
    dates_from_variable,
)
from LSTM.training import load_ensemble_bundle as load_lstm_bundle


PRODUCT_VARIABLES = {"ASCAT": "ASCAT", "SMAP": "SMAP"}
MODEL_NAMES = ("EF", "FNO", "LSTM")
DEFAULT_BOUNDS = (-126.0, -66.0, 24.0, 51.0)
DEFAULT_PIXEL_BATCH = int(
    os.environ.get("CONUS_PREDICTION_PIXEL_BATCH", "32")
)
DEFAULT_INFERENCE_BATCH = int(
    os.environ.get("CONUS_PREDICTION_INFERENCE_BATCH", "4096")
)


def resolve_prediction_batch_sizes(
    device,
    pixel_batch=None,
    inference_batch=None,
):
    """Choose conservative neural batches for the available device memory.

    Explicit arguments take precedence over environment overrides. When no
    override is supplied, CUDA devices use larger batches than the CPU, with
    an additional increase at 20 and 40 GiB of device memory. FNO and LSTM use
    the same settings so their spatial support and execution policy match.
    """
    device = torch.device(device)
    total_gib = None
    if device.type == "cuda":
        total_memory = torch.cuda.get_device_properties(device).total_memory
        total_gib = total_memory / (1024**3)

    if pixel_batch is None:
        pixel_batch = DEFAULT_PIXEL_BATCH
        if (
            "CONUS_PREDICTION_PIXEL_BATCH" not in os.environ
            and device.type == "cuda"
        ):
            pixel_batch = 128 if total_gib >= 40 else 64

    if inference_batch is None:
        inference_batch = DEFAULT_INFERENCE_BATCH
        if (
            "CONUS_PREDICTION_INFERENCE_BATCH" not in os.environ
            and device.type == "cuda"
        ):
            if total_gib >= 40:
                inference_batch = 32768
            elif total_gib >= 20:
                inference_batch = 8192

    pixel_batch = int(pixel_batch)
    inference_batch = int(inference_batch)
    if pixel_batch < 1:
        raise ValueError("pixel_batch must be positive.")
    if inference_batch < 1:
        raise ValueError("inference_batch must be positive.")
    return pixel_batch, inference_batch


def _filled(values, dtype=np.float32):
    return np.asarray(np.ma.filled(values, np.nan), dtype=dtype)


def _coordinate_name(dataset, candidates):
    for name in candidates:
        if name in dataset.variables:
            return name
    raise KeyError(f"Missing coordinate; tried {', '.join(candidates)}")


def _coordinate_axis(dataset, name, axis):
    values = _filled(dataset.variables[name][:])
    if values.ndim == 1:
        return values
    if values.ndim != 2:
        raise ValueError(f"{name!r} must be a 1-D or regular 2-D coordinate.")
    first_column = values[:, 0]
    first_row = values[0, :]
    if axis == "lat":
        if np.allclose(values, first_column[:, None], equal_nan=True):
            return first_column
        if np.allclose(values, first_row[None, :], equal_nan=True):
            return first_row
    else:
        if np.allclose(values, first_row[None, :], equal_nan=True):
            return first_row
        if np.allclose(values, first_column[:, None], equal_nan=True):
            return first_column
    raise ValueError(f"{name!r} is not a regular latitude/longitude grid.")


def _regional_slice(latitude, longitude, bounds):
    lon_min, lon_max, lat_min, lat_max = (float(value) for value in bounds)
    latitude_indices = np.flatnonzero(
        (latitude >= lat_min) & (latitude <= lat_max)
    )
    longitude_indices = np.flatnonzero(
        (longitude >= lon_min) & (longitude <= lon_max)
    )
    if not len(latitude_indices) or not len(longitude_indices):
        raise ValueError(f"No cells occur inside bounds {tuple(bounds)}.")
    return (
        slice(int(latitude_indices.min()), int(latitude_indices.max()) + 1),
        slice(int(longitude_indices.min()), int(longitude_indices.max()) + 1),
    )


def _axis_flip(reference, candidate, label, tolerance=1e-4):
    if reference.shape != candidate.shape:
        raise ValueError(
            f"{label} coordinate shapes differ: {reference.shape} and "
            f"{candidate.shape}."
        )
    if np.allclose(reference, candidate, atol=tolerance, rtol=0.0):
        return False
    if np.allclose(reference, candidate[::-1], atol=tolerance, rtol=0.0):
        return True
    raise ValueError(
        f"{label} coordinates do not align with "
        "Model_input_static_eqd_010.nc."
    )


def _read_product_chunk(
    dataset,
    variable_name,
    latitude_name,
    longitude_name,
    latitude_slice,
    longitude_slice,
    time_indices,
):
    variable = dataset.variables[variable_name]
    selections = []
    retained_axes = {}
    for dimension in variable.dimensions:
        if dimension == latitude_name:
            retained_axes["lat"] = len(selections)
            selections.append(latitude_slice)
        elif dimension == longitude_name:
            retained_axes["lon"] = len(selections)
            selections.append(longitude_slice)
        elif dimension == "time":
            retained_axes["time"] = len(selections)
            if len(time_indices) and np.all(np.diff(time_indices) == 1):
                selections.append(
                    slice(int(time_indices[0]), int(time_indices[-1]) + 1)
                )
            else:
                selections.append(time_indices)
        else:
            selections.append(0)
    if set(retained_axes) != {"lat", "lon", "time"}:
        raise ValueError(
            f"{variable_name!r} must have latitude, longitude, and time axes."
        )
    values = _filled(variable[tuple(selections)])
    return np.transpose(
        values,
        (
            retained_axes["lat"],
            retained_axes["lon"],
            retained_axes["time"],
        ),
    )


def _clean_ssm(values):
    values = np.asarray(values, dtype=np.float32)
    return np.where(
        np.isfinite(values) & (values > 0.0) & (values <= 1.0),
        values,
        np.nan,
    ).astype(np.float32)


def _models_from_bundle(model_name, bundle, device):
    parameters = bundle["hyperparameters"]
    feature_count = len(bundle["scaler"]["feature_names"])
    models = []
    for state_dict in bundle["state_dicts"]:
        if model_name == "FNO":
            model = FNO(
                modes=int(parameters["modes"]),
                width=int(parameters["width"]),
                num_static_properties=feature_count,
                dropout_static=float(parameters.get("dropout_static", 0.0)),
                dropout_fc=float(parameters.get("dropout_fc", 0.0)),
            )
        elif model_name == "LSTM":
            model = LSTM(
                hidden_size=int(parameters.get("hidden_size", 64)),
                num_static_properties=feature_count,
                num_layers=int(parameters.get("num_layers", 2)),
                dropout_static=float(parameters.get("dropout_static", 0.0)),
                dropout_fc=float(parameters.get("dropout_fc", 0.0)),
            )
        else:
            raise ValueError(f"Unsupported neural model {model_name!r}.")
        model.load_state_dict(state_dict)
        model.to(device).eval()
        models.append(model)
    return models


def _load_neural_model(model_name, bundle_file, device):
    if model_name == "FNO":
        bundle = load_fno_bundle(bundle_file)
    elif model_name == "LSTM":
        bundle = load_lstm_bundle(bundle_file)
    else:
        raise ValueError(f"Unsupported neural model {model_name!r}.")
    feature_names = list(bundle["scaler"]["feature_names"])
    if feature_names != list(STATIC_FEATURES):
        raise ValueError(
            f"{model_name} bundle does not use the required seven static features."
        )
    return {
        "bundle": bundle,
        "models": _models_from_bundle(model_name, bundle, device),
    }


def _static_chunk(dataset, latitude_slice, longitude_slice, scaler):
    feature_names = list(scaler["feature_names"])
    missing = [name for name in feature_names if name not in dataset.variables]
    if missing:
        raise KeyError(
            "Model_input_static_eqd_010.nc is missing static variables: "
            f"{missing}"
        )
    raw = np.stack(
        [
            _filled(dataset.variables[name][latitude_slice, longitude_slice])
            for name in feature_names
        ],
        axis=-1,
    ).reshape(-1, len(feature_names))
    mean = np.asarray(scaler["mean"], dtype=np.float32)
    standard_deviation = np.asarray(
        scaler["standard_deviation"], dtype=np.float32
    )
    scaled = np.zeros_like(raw)
    np.divide(
        raw - mean,
        standard_deviation,
        out=scaled,
        where=standard_deviation != 0,
    )
    valid = np.all(np.isfinite(raw), axis=1) & np.all(
        np.isfinite(scaled), axis=1
    )
    return scaled.astype(np.float32), valid


def _neural_prediction(
    models,
    ssm,
    static,
    static_valid,
    dates,
    device,
    pixel_batch,
    inference_batch,
):
    mean_grid = np.full(ssm.shape, np.nan, dtype=np.float32)
    standard_deviation_grid = np.full(ssm.shape, np.nan, dtype=np.float32)
    valid_pixels = np.flatnonzero(static_valid)

    for pixel_start in range(0, len(valid_pixels), int(pixel_batch)):
        pixel_indices = valid_pixels[pixel_start : pixel_start + int(pixel_batch)]
        windows = []
        owners = []
        targets = []
        for pixel_index in pixel_indices:
            dynamic, _, target_indices = build_observation_sequences(
                ssm[pixel_index],
                time_values=dates,
                target_start=0,
                target_stop=len(dates),
                window_size=WINDOW_SIZE,
            )
            if not len(dynamic):
                continue
            windows.append(dynamic)
            owners.append(
                np.full(len(dynamic), pixel_index, dtype=np.int64)
            )
            targets.append(target_indices)
        if not windows:
            continue

        dynamic = np.concatenate(windows).astype(np.float32)
        owner = np.concatenate(owners)
        target = np.concatenate(targets)
        for event_start in range(0, len(dynamic), int(inference_batch)):
            event_stop = min(event_start + int(inference_batch), len(dynamic))
            event_owner = owner[event_start:event_stop]
            dynamic_tensor = torch.as_tensor(
                dynamic[event_start:event_stop], dtype=torch.float32, device=device
            )
            static_tensor = torch.as_tensor(
                static[event_owner], dtype=torch.float32, device=device
            )
            trials = []
            with torch.inference_mode():
                for model in models:
                    trials.append(
                        model(dynamic_tensor, static_tensor)
                        .reshape(-1)
                        .detach()
                        .cpu()
                        .numpy()
                    )
            trials = np.asarray(trials, dtype=np.float32)
            event_mean = trials.mean(axis=0)
            event_standard_deviation = trials.std(axis=0, ddof=1)
            event_target = target[event_start:event_stop]
            mean_grid[event_owner, event_target] = event_mean
            standard_deviation_grid[event_owner, event_target] = (
                event_standard_deviation
            )
    return mean_grid, standard_deviation_grid


def _create_output(
    path,
    model_name,
    product,
    latitude,
    longitude,
    source_time,
    time_indices,
    include_standard_deviation,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists():
        temporary.unlink()
    output = nc.Dataset(temporary, "w", format="NETCDF4")
    output.createDimension("lat", len(latitude))
    output.createDimension("lon", len(longitude))
    output.createDimension("time", len(time_indices))

    lat_variable = output.createVariable("lat", "f4", ("lat",))
    lon_variable = output.createVariable("lon", "f4", ("lon",))
    time_variable = output.createVariable(
        "time", source_time.datatype, ("time",)
    )
    lat_variable[:] = latitude
    lon_variable[:] = longitude
    time_variable[:] = np.asarray(source_time[:])[time_indices]
    lat_variable.units = "degrees_north"
    lon_variable.units = "degrees_east"
    for attribute in source_time.ncattrs():
        if attribute != "_FillValue":
            time_variable.setncattr(attribute, source_time.getncattr(attribute))

    chunks = (1, min(300, len(longitude)), min(128, len(time_indices)))
    prediction = output.createVariable(
        "RZSM_prediction",
        "f4",
        ("lat", "lon", "time"),
        zlib=True,
        complevel=4,
        chunksizes=chunks,
        fill_value=np.float32(np.nan),
    )
    prediction.units = "m3 m-3"
    prediction.long_name = f"{model_name} RZSM prediction from {product} SSM"
    if include_standard_deviation:
        spread = output.createVariable(
            "RZSM_prediction_std",
            "f4",
            ("lat", "lon", "time"),
            zlib=True,
            complevel=4,
            chunksizes=chunks,
            fill_value=np.float32(np.nan),
        )
        spread.units = "m3 m-3"
        spread.long_name = "Sample standard deviation across model seeds"
        spread.ddof = 1
    return output, temporary, path


def run_conus_prediction(
    *,
    product,
    product_file,
    static_file,
    output_directory,
    models=("EF", "FNO", "LSTM"),
    fno_bundle_file=None,
    lstm_bundle_file=None,
    bounds=DEFAULT_BOUNDS,
    ef_time_constant_days=15.0,
    ef_spinup_days=365,
    chunk_latitude=10,
    pixel_batch=None,
    inference_batch=None,
    device=None,
):
    """Create one final prediction file per requested model for one SSM product.

    The complete 2015-04-01--2023-12-31 context is read. EF values before
    365 elapsed days are masked, whereas neural output begins independently at
    each pixel's 32nd finite surface-observation event.
    """
    product = str(product).upper()
    if product not in PRODUCT_VARIABLES:
        raise ValueError(f"Unsupported product {product!r}.")
    models = tuple(str(model).upper() for model in models)
    invalid_models = sorted(set(models).difference(MODEL_NAMES))
    if invalid_models:
        raise ValueError(f"Unsupported models: {invalid_models}")
    if len(set(models)) != len(models):
        raise ValueError("Each model may be requested only once.")
    if int(chunk_latitude) < 1:
        raise ValueError("chunk_latitude must be positive.")

    device = torch.device(
        device
        if device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    neural_models = tuple(name for name in models if name in {"FNO", "LSTM"})
    if neural_models:
        pixel_batch, inference_batch = resolve_prediction_batch_sizes(
            device,
            pixel_batch=pixel_batch,
            inference_batch=inference_batch,
        )
    neural = {}
    if "FNO" in models:
        if fno_bundle_file is None:
            raise ValueError("fno_bundle_file is required for FNO prediction.")
        neural["FNO"] = _load_neural_model("FNO", fno_bundle_file, device)
    if "LSTM" in models:
        if lstm_bundle_file is None:
            raise ValueError("lstm_bundle_file is required for LSTM prediction.")
        neural["LSTM"] = _load_neural_model("LSTM", lstm_bundle_file, device)

    product_file = Path(product_file)
    static_file = Path(static_file)
    if not product_file.is_file():
        raise FileNotFoundError(product_file)
    if not static_file.is_file():
        raise FileNotFoundError(static_file)
    output_directory = Path(output_directory)

    outputs = {}
    try:
        with nc.Dataset(product_file) as product_dataset, nc.Dataset(
            static_file
        ) as static_dataset:
            variable_name = PRODUCT_VARIABLES[product]
            if variable_name not in product_dataset.variables:
                raise KeyError(
                    f"{product_file} has no {variable_name!r} variable."
                )
            if "CONUS_mask" not in static_dataset.variables:
                raise KeyError(f"{static_file} has no 'CONUS_mask' variable.")

            product_latitude_name = _coordinate_name(
                product_dataset, ("lat", "latitude")
            )
            product_longitude_name = _coordinate_name(
                product_dataset, ("lon", "longitude")
            )
            static_latitude_name = _coordinate_name(
                static_dataset, ("lat", "latitude")
            )
            static_longitude_name = _coordinate_name(
                static_dataset, ("lon", "longitude")
            )
            product_latitude = _coordinate_axis(
                product_dataset, product_latitude_name, "lat"
            )
            product_longitude = _coordinate_axis(
                product_dataset, product_longitude_name, "lon"
            )
            static_latitude = _coordinate_axis(
                static_dataset, static_latitude_name, "lat"
            )
            static_longitude = _coordinate_axis(
                static_dataset, static_longitude_name, "lon"
            )
            static_latitude_slice, static_longitude_slice = _regional_slice(
                static_latitude, static_longitude, bounds
            )
            product_latitude_slice, product_longitude_slice = _regional_slice(
                product_latitude, product_longitude, bounds
            )
            latitude = static_latitude[static_latitude_slice]
            longitude = static_longitude[static_longitude_slice]
            flip_latitude = _axis_flip(
                latitude,
                product_latitude[product_latitude_slice],
                f"{product} latitude",
            )
            flip_longitude = _axis_flip(
                longitude,
                product_longitude[product_longitude_slice],
                f"{product} longitude",
            )

            source_time = product_dataset.variables["time"]
            all_dates = dates_from_variable(source_time)
            time_indices = context_period_indices(all_dates, require_complete=True)
            dates = all_dates[time_indices]
            if dates[0] != CONTEXT_START_DATE or dates[-1] != CONTEXT_END_DATE:
                raise ValueError("Unexpected model context dates.")
            time_days = (
                (dates - dates[0]) / np.timedelta64(1, "D")
            ).astype(np.float64)

            for model_name in models:
                output_model_name = "Fixed_EF" if model_name == "EF" else model_name
                path = output_directory / f"{output_model_name}_{product}_prediction.nc"
                outputs[model_name] = _create_output(
                    path,
                    model_name,
                    product,
                    latitude,
                    longitude,
                    source_time,
                    time_indices,
                    include_standard_deviation=model_name in neural,
                )
                output = outputs[model_name][0]
                output.latitude_chunk_size = int(chunk_latitude)
                if model_name in neural:
                    output.pixel_batch_size = int(pixel_batch)
                    output.inference_batch_size = int(inference_batch)
                    output.neural_batch_scope = (
                        "shared FNO/LSTM policy; inference batch is the maximum "
                        "event count per seeded-model forward pass"
                    )
                else:
                    output.batch_policy = (
                        "EF is vectorized over every pixel in one latitude chunk; "
                        "neural pixel and inference batches do not apply"
                    )

            latitude_count = len(latitude)
            for local_start in tqdm(
                range(0, latitude_count, int(chunk_latitude)),
                desc=f"{product} CONUS prediction",
                unit="chunk",
            ):
                local_stop = min(
                    local_start + int(chunk_latitude), latitude_count
                )
                static_chunk_latitude = slice(
                    static_latitude_slice.start + local_start,
                    static_latitude_slice.start + local_stop,
                )
                if flip_latitude:
                    product_local_start = latitude_count - local_stop
                    product_local_stop = latitude_count - local_start
                else:
                    product_local_start = local_start
                    product_local_stop = local_stop
                product_chunk_latitude = slice(
                    product_latitude_slice.start + product_local_start,
                    product_latitude_slice.start + product_local_stop,
                )
                ssm_grid = _read_product_chunk(
                    product_dataset,
                    variable_name,
                    product_latitude_name,
                    product_longitude_name,
                    product_chunk_latitude,
                    product_longitude_slice,
                    time_indices,
                )
                if flip_latitude:
                    ssm_grid = ssm_grid[::-1]
                if flip_longitude:
                    ssm_grid = ssm_grid[:, ::-1]
                ssm_grid = _clean_ssm(ssm_grid)
                ssm = ssm_grid.reshape(-1, len(dates))
                conus_mask = np.asarray(
                    static_dataset.variables["CONUS_mask"][
                        static_chunk_latitude, static_longitude_slice
                    ],
                    dtype=bool,
                ).reshape(-1)
                ssm[~conus_mask] = np.nan

                if "EF" in models:
                    ef_prediction = exponential_filter(
                        ssm, time_days, ef_time_constant_days
                    )
                    ef_prediction[:, time_days < float(ef_spinup_days)] = np.nan
                    ef_prediction[~conus_mask] = np.nan
                    outputs["EF"][0].variables["RZSM_prediction"][
                        local_start:local_stop
                    ] = ef_prediction.reshape(ssm_grid.shape)

                for model_name, model_context in neural.items():
                    static, static_valid = _static_chunk(
                        static_dataset,
                        static_chunk_latitude,
                        static_longitude_slice,
                        model_context["bundle"]["scaler"],
                    )
                    static_valid &= conus_mask
                    prediction, standard_deviation = _neural_prediction(
                        model_context["models"],
                        ssm,
                        static,
                        static_valid,
                        dates,
                        device,
                        pixel_batch,
                        inference_batch,
                    )
                    output = outputs[model_name][0]
                    output.variables["RZSM_prediction"][local_start:local_stop] = (
                        prediction.reshape(ssm_grid.shape)
                    )
                    output.variables["RZSM_prediction_std"][
                        local_start:local_stop
                    ] = standard_deviation.reshape(ssm_grid.shape)

        final_paths = {}
        for model_name, (dataset, temporary, final) in outputs.items():
            dataset.close()
            os.replace(temporary, final)
            final_paths[model_name] = final
        return final_paths
    except Exception:
        for dataset, temporary, _ in outputs.values():
            try:
                dataset.close()
            except Exception:
                pass
            if temporary.exists():
                temporary.unlink()
        raise
