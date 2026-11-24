'''Estimate CONUS RZSM with the exponential filter for SMAP and ASCAT.

Steps 03 and 04 reuse this script's yearly estimation and NetCDF merging.
'''

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PUBLICATION_ROOT = Path(__file__).resolve().parents[1]
if str(PUBLICATION_ROOT) not in sys.path:
    sys.path.insert(0, str(PUBLICATION_ROOT))

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import netCDF4 as nc
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from core import static_input_profiles as static_profiles
from core.FNOEvent import FNOEvent
from core.LSTMEvent import LSTMEvent
from core.precipitation_inputs import (
    EVENT_PRECIPITATION_CHANNELS,
    RAINFALL_FILE,
    RAINFALL_VARIABLE,
    event_interval_amounts,
    extract_daily_pixels,
    rainfall_path,
    require_rainfall_file,
    standardized_precipitation_windows,
)
from core.project_paths import DATA_ROOT, PATHS, estimation_output_root, existing_result_path, project_results_root
from core.rzsm_pipeline_config import (
    ASCAT_PRIMARY_FILENAME,
    COMMON_END_DATE,
    DEFAULT_FNO_MODEL,
    DEFAULT_LSTM_MODEL,
    ESTIMATION_YEARS,
    EF_POLICY,
    EF_SPIN_UP_DAYS,
    FNO_MODEL_CHOICES,
    LSTM_MODEL_CHOICES,
    MODEL_SEEDS,
    NEURAL_MODEL_CHOICES,
    NEURAL_OUTPUT_START_DATE,
    NEURAL_RECIPE_EVENT,
    OBSERVATION_EVENT_GAP_SCALE_DAYS,
    OBSERVATION_EVENT_SEQUENCE_LENGTH,
    ESTIMATION_START_DATE,
    build_observation_event_sequences,
    load_ef_t,
    neural_development_metadata,
    neural_model_family,
    observation_event_support_mask,
    prepare_observation_time_days,
)


YEARS = list(range(2015, 2024))
BOUNDS = (-126.0, -66.0, 24.0, 51.0)
ACTIVE_STATIC_PROFILE = static_profiles.EVENT_PROFILE
ACTIVE_EF_T = None

MODEL_SAVE_MAP = {
    'EF': 'Exponential_Filter',
    **{name: 'Fourier_Neural_Operator' for name in FNO_MODEL_CHOICES},
    **{name: 'Long_Short-Term_Memory' for name in LSTM_MODEL_CHOICES},
}

PRODUCTS = {
    'ASCAT': {
        'path_parts': ('Advanced_Scatterometer', ASCAT_PRIMARY_FILENAME),
        'variable': 'ASCAT_GLDAS',
        'valid_min': 0.0,
        'minimum_inclusive': True,
    },
    'SMAP': {
        'path_parts': ('Soil_Moisture_Active_Passive', 'SPL3SMP_E.006_2015_2023_R_eqd_010.nc'),
        'variable': 'SMAP',
    },
}

CHUNK_LAT = 10
PIXEL_BATCH = 32
INFERENCE_BATCH = 4096
DEFAULT_WORKERS = min(8, max(1, os.cpu_count() or 1))
ESTIMATION_ROOT_OVERRIDE = None


MERGE_CHUNK_LAT = 25
CONSISTENT_ATTRIBUTES = (
    'EF_filter_T_days',
    'model_seeds',
    'trial_count',
    'static_schema_version',
    'static_features',
    'dynamic_channels',
    'sequence_length_observations',
)


def validate_gpus(gpus):
    '''Select devices for neural estimation.'''
    cuda = torch.cuda
    gpu_count = cuda.device_count() if cuda.is_available() else 0

    if gpus is None:
        gpus = list(range(min(gpu_count, len(MODEL_SEEDS))))
        if not gpus:
            print('No CUDA device is visible; FNO/LSTM estimation runs on CPU.')
        return gpus

    gpus = [int(gpu) for gpu in gpus]

    if len(set(gpus)) != len(gpus):
        raise ValueError(f'Duplicate GPU indices requested: {gpus}.')

    if len(gpus) > len(MODEL_SEEDS):
        raise ValueError(
            f'{len(gpus)} GPUs requested, but the ensemble has only '
            f'{len(MODEL_SEEDS)} seeded models.'
        )

    if gpus and not gpu_count:
        raise RuntimeError(
            'CUDA is not available; omit --gpus or use --gpus cpu.'
        )

    for gpu in gpus:
        if gpu < 0 or gpu >= gpu_count:
            raise ValueError(
                f'Invalid GPU index {gpu}; available GPUs are '
                f'0-{gpu_count - 1}.'
            )

    return gpus


def resolve_estimation_devices(gpus):
    '''Resolve CUDA devices with a CPU fallback.'''
    gpus = [int(gpu) for gpu in (gpus or [])]
    if not gpus or not torch.cuda.is_available():
        return [torch.device('cpu')]
    return [torch.device(f'cuda:{gpu}') for gpu in gpus]


def assign_trial_devices(model_count, device_count):
    '''Split the seeded trials into contiguous, near-equal per-device blocks.

    Ten trials over two devices give ``[[0..4], [5..9]]``.
    '''
    if model_count < 1 or device_count < 1:
        raise ValueError('At least one seeded trial and one device are required.')
    if device_count > model_count:
        raise ValueError(f'{device_count} devices exceed the {model_count} seeded trials.')
    base, extra = divmod(int(model_count), int(device_count))
    assignments, start = [], 0
    for index in range(device_count):
        owned = base + int(index < extra)
        assignments.append(list(range(start, start + owned)))
        start += owned
    return assignments


def expand_trial_devices(model_count, devices):
    '''Map seeded trials to devices in checkpoint order.'''
    devices = list(devices)
    owners = [None] * int(model_count)
    for device, indices in zip(devices, assign_trial_devices(model_count, len(devices))):
        for index in indices:
            owners[index] = device
    return owners


def resolve_estimation_batch_sizes(devices):
    '''Choose pixel and inference batch sizes from the smallest GPU's memory.'''
    cuda_devices = [device for device in devices if device.type == 'cuda']
    if not cuda_devices:
        return PIXEL_BATCH, INFERENCE_BATCH
    total_gib = min(
        torch.cuda.get_device_properties(device).total_memory for device in cuda_devices
    ) / (1024 ** 3)
    pixel_batch = 128 if total_gib >= 40 else 64
    if total_gib >= 40:
        inference_batch = 32768
    elif total_gib >= 20:
        inference_batch = 8192
    else:
        inference_batch = INFERENCE_BATCH
    return pixel_batch, inference_batch


def estimation_root():
    if ESTIMATION_ROOT_OVERRIDE is not None:
        return ESTIMATION_ROOT_OVERRIDE
    return os.fspath(estimation_output_root(DATA_ROOT))


def model_output_dir(model_name):
    return os.path.join(estimation_root(), MODEL_SAVE_MAP[model_name])


def yearly_output_path(model_name, data, year):
    return os.path.join(
        model_output_dir(model_name), 'Yearly', f'{model_name}_{data}_{year}_estimation.nc'
    )


def product_path(data):
    return os.path.join(project_results_root(DATA_ROOT), *PRODUCTS[data]['path_parts'])


def static_input_path():
    return os.fspath(static_profiles.grid_path(DATA_ROOT, ACTIVE_STATIC_PROFILE, read_existing=True))


def rainfall_input_path():
    return os.fspath(rainfall_path(DATA_ROOT))


def coord_name(ds, candidates):
    for name in candidates:
        if name in ds.variables:
            return name
    raise KeyError(f"Missing coordinate. Tried: {', '.join(candidates)}")


def coord_axis_1d(ds, name, axis_label):
    '''Extract a coordinate axis from a vector or regular mesh.'''
    values = np.ma.filled(ds.variables[name][:], np.nan).astype(np.float32)
    if values.ndim == 1:
        return values
    if values.ndim == 2:
        first_col, first_row = values[:, 0], values[0, :]
        candidates = (
            ((first_col, first_col[:, None]), (first_row, first_row[None, :]))
            if axis_label == 'lat'
            else ((first_row, first_row[None, :]), (first_col, first_col[:, None]))
        )
        for axis, mesh in candidates:
            if np.allclose(values, mesh, equal_nan=True):
                return axis
    raise ValueError(
        f'Coordinate variable {name!r} must be 1D or a regular 2D grid; got shape {values.shape}.'
    )


def fill_nan(values):
    return np.ma.filled(values, np.nan).astype(np.float32)


def conus_slices(lat_1d, lon_1d):
    lon_min, lon_max, lat_min, lat_max = BOUNDS
    lat_idx = np.where((lat_1d >= lat_min) & (lat_1d <= lat_max))[0]
    lon_idx = np.where((lon_1d >= lon_min) & (lon_1d <= lon_max))[0]
    if len(lat_idx) == 0 or len(lon_idx) == 0:
        raise ValueError(f'No grid cells found inside bounds {BOUNDS}.')
    return (
        slice(int(lat_idx.min()), int(lat_idx.max()) + 1),
        slice(int(lon_idx.min()), int(lon_idx.max()) + 1),
    )


def axis_flip_to_reference(reference, target, label, atol=1e-4):
    '''Detect reversed coordinate axes and reject other mismatches.'''
    reference, target = np.asarray(reference), np.asarray(target)
    if reference.shape != target.shape:
        raise ValueError(f'{label} axis shape mismatch: {reference.shape} vs {target.shape}')
    if np.nanmax(np.abs(reference - target)) <= atol:
        return False
    if np.nanmax(np.abs(reference - target[::-1])) <= atol:
        return True
    raise ValueError(f'{label} axis values do not align between Model_input and product grid.')


def read_dates(time_var):
    dates = nc.num2date(
        np.asarray(time_var[:]),
        units=time_var.units,
        calendar=getattr(time_var, 'calendar', 'standard'),
        only_use_cftime_datetimes=False,
        only_use_python_datetimes=True,
    )
    return np.asarray(
        [np.datetime64(f'{d.year:04d}-{d.month:02d}-{d.day:02d}', 'D') for d in dates],
        dtype='datetime64[D]',
    )


def selected_time_info(product_ds, years):
    '''Select the full daily record of ``years`` and the positions written per year.

    The whole record from ``NEURAL_OUTPUT_START_DATE`` is kept for EF state and
    event windows; only dates from ``ESTIMATION_START_DATE`` are written.
    '''
    time_var = product_ds.variables['time']
    raw_time = np.asarray(time_var[:])
    dates = read_dates(time_var)
    year_values = dates.astype('datetime64[Y]').astype(int) + 1970
    selected_idx = np.where(
        np.isin(year_values, years)
        & (dates >= NEURAL_OUTPUT_START_DATE)
        & (dates <= COMMON_END_DATE)
    )[0]
    if len(selected_idx) == 0:
        raise ValueError(f'No selected years {years} were found in the product time axis.')

    selected_dates = dates[selected_idx]
    expected = np.arange(
        NEURAL_OUTPUT_START_DATE, COMMON_END_DATE + np.timedelta64(1, 'D'), dtype='datetime64[D]'
    )
    expected = expected[np.isin(expected.astype('datetime64[Y]').astype(int) + 1970, years)]
    if not np.array_equal(selected_dates, expected):
        missing = expected[~np.isin(expected, selected_dates)]
        detail = f', first missing={missing[0]}' if len(missing) else ''
        raise ValueError(
            f'Input product does not provide a complete daily calendar for {years}: '
            f'expected {len(expected)} days, found {len(selected_dates)}{detail}.'
        )
    selected_years = year_values[selected_idx]
    output_mask = selected_dates >= ESTIMATION_START_DATE
    year_positions = {
        int(year): np.where((selected_years == int(year)) & output_mask)[0]
        for year in years
        if np.any((selected_years == int(year)) & output_mask)
    }
    time_days = (selected_dates - selected_dates[0]).astype('timedelta64[D]').astype(np.float32)
    time_attrs = {
        attr: time_var.getncattr(attr) for attr in time_var.ncattrs() if attr != '_FillValue'
    }
    return selected_idx, raw_time[selected_idx], selected_dates, time_days, year_positions, time_attrs


def read_product_chunk(product_ds, variable_name, lat_slice, lon_slice, time_idx, lat_name, lon_name):
    '''Read one (lat, lon, time) block whatever the variable's dimension order.'''
    var = product_ds.variables[variable_name]
    index, axis_order = [], {}
    for dim in var.dimensions:
        if dim == lat_name:
            axis_order['lat'] = len(index)
            index.append(lat_slice)
        elif dim == lon_name:
            axis_order['lon'] = len(index)
            index.append(lon_slice)
        elif dim == 'time':
            axis_order['time'] = len(index)
            if len(time_idx) and np.all(np.diff(time_idx) == 1):
                index.append(slice(int(time_idx[0]), int(time_idx[-1]) + 1))
            else:
                index.append(time_idx)
        else:
            index.append(0)
    data = fill_nan(var[tuple(index)])
    return np.transpose(data, (axis_order['lat'], axis_order['lon'], axis_order['time']))


def clean_ssm(values, data):
    '''Apply the source product's physical range before EF/model inference.'''
    settings = PRODUCTS[data]
    valid_min = float(settings.get('valid_min', 0.0))
    valid_max = float(settings.get('valid_max', 1.0))
    if settings.get('minimum_inclusive', False):
        lower_valid = values >= valid_min
    else:
        lower_valid = values > valid_min
    return np.where(lower_valid & (values <= valid_max), values, np.nan).astype(np.float32)


def read_static_chunk(static_ds, lat_slice, lon_slice, scaler_df):
    '''Read and standardize static predictors in scaler order.'''
    feature_cols = scaler_df['Feature'].tolist()
    missing = [feature for feature in feature_cols if feature not in static_ds.variables]
    if missing:
        raise KeyError(f'Static features {missing} are missing from {static_input_path()}')
    raw = np.stack(
        [fill_nan(static_ds.variables[feature][lat_slice, lon_slice]) for feature in feature_cols],
        axis=-1,
    )
    scaled = raw.copy()
    for i, row in scaler_df.reset_index(drop=True).iterrows():
        std = row['Std']
        if pd.notna(std) and std != 0:
            scaled[..., i] = (scaled[..., i] - row['Mean']) / std
        else:
            scaled[..., i] = 0.0
    raw_2d = raw.reshape(-1, raw.shape[-1])
    scaled_2d = scaled.reshape(-1, scaled.shape[-1])
    static_valid = np.all(np.isfinite(raw_2d), axis=1) & np.all(np.isfinite(scaled_2d), axis=1)
    return raw_2d.astype(np.float32), scaled_2d.astype(np.float32), static_valid, feature_cols


def create_trial_parallel_backend(models, devices):
    '''Group the seeded trials by device so they can run concurrently.

    Trials are spread over ``devices`` in contiguous blocks. On a GPU every
    trial gets its own CUDA stream; on the CPU a thread pool runs them.
    '''
    groups = []
    for device, indices in zip(devices, assign_trial_devices(len(models), len(devices))):
        group_models = [models[index] for index in indices]
        streams = executor = None
        if len(group_models) > 1:
            if device.type == 'cuda':
                streams = [torch.cuda.Stream(device=device) for _ in group_models]
            else:
                executor = ThreadPoolExecutor(max_workers=len(group_models))
        groups.append({
            'device': device,
            'model_indices': indices,
            'models': group_models,
            'trial_streams': streams,
            'trial_executor': executor,
        })
    return groups


def model_forward_no_grad(model, x_dynamic, x_static):
    with torch.inference_mode():
        return model(x_dynamic, x_static)


def simultaneous_trial_forward(models, x_dynamic, x_static, device, trial_executor=None, trial_streams=None):
    '''Evaluate one device's seeded models concurrently on the same inputs.'''
    if device.type == 'cuda' and trial_streams is not None:
        with torch.cuda.device(device):
            current_stream = torch.cuda.current_stream(device=device)
            for stream in trial_streams:
                stream.wait_stream(current_stream)
                x_dynamic.record_stream(stream)
                x_static.record_stream(stream)
            estimations = []
            with torch.inference_mode():
                for model, stream in zip(models, trial_streams):
                    with torch.cuda.stream(stream):
                        estimations.append(model(x_dynamic, x_static))
            for stream in trial_streams:
                current_stream.wait_stream(stream)
        return estimations
    if trial_executor is not None:
        futures = [
            trial_executor.submit(model_forward_no_grad, model, x_dynamic, x_static)
            for model in models
        ]
        return [future.result() for future in futures]
    return [model_forward_no_grad(model, x_dynamic, x_static) for model in models]


def ensemble_inference_sequence(models, x_windows, x_static, trial_device_groups, inference_batch=INFERENCE_BATCH):
    '''Run the seeded neural models over the event windows.'''
    per_trial_batch = max(
        1, int(inference_batch) // max(len(group['models']) for group in trial_device_groups)
    )
    output_batches = []
    for start in range(0, x_windows.shape[0], per_trial_batch):
        end = min(start + per_trial_batch, x_windows.shape[0])
        pending = []
        for group in trial_device_groups:
            device = group['device']
            x_dyn = torch.tensor(x_windows[start:end], dtype=torch.float32, device=device)
            x_stat = torch.tensor(x_static[start:end], dtype=torch.float32, device=device)
            estimations = simultaneous_trial_forward(
                group['models'], x_dyn, x_stat, device,
                trial_executor=group.get('trial_executor'),
                trial_streams=group.get('trial_streams'),
            )
            pending.append((group['model_indices'], torch.stack([p.squeeze(-1) for p in estimations])))

        batch_output = np.empty((len(models), end - start), dtype=np.float32)
        for model_indices, stacked in pending:
            host = stacked.detach().cpu().numpy().astype(np.float32, copy=False)
            for row, model_index in enumerate(model_indices):
                batch_output[model_index] = host[row]
        output_batches.append(batch_output)

    combined = np.concatenate(output_batches, axis=1)
    return [combined[index] for index in range(len(models))]


def build_pixel_estimation_windows(task):
    '''Build one pixel's previous-observation event windows (run in a worker thread).'''
    pixel_index, ssm_series, prepared_time_days, target_start, target_stop = task
    windows, _, target_indices = build_observation_event_sequences(
        ssm_series,
        prepared_time_days=prepared_time_days,
        target_start=target_start,
        target_stop=target_stop,
        window_size=OBSERVATION_EVENT_SEQUENCE_LENGTH,
        gap_scale_days=OBSERVATION_EVENT_GAP_SCALE_DAYS,
    )
    return pixel_index, windows, target_indices


def summarize_trial_estimations(trial_estimations):
    '''Summarize trial estimates by date.'''
    estimations = np.asarray(trial_estimations, dtype=np.float64)
    if estimations.ndim != 2:
        raise ValueError(
            "Trial estimations must have shape (number_of_trials, number_of_times)."
        )

    finite = np.isfinite(estimations)
    counts = finite.sum(axis=0)
    sums = np.where(finite, estimations, 0.0).sum(axis=0)

    mean = np.full(estimations.shape[1], np.nan, dtype=np.float64)
    np.divide(sums, counts, out=mean, where=counts > 0)

    centered = np.where(finite, estimations - mean, 0.0)
    squared_deviations = np.square(centered).sum(axis=0)
    variance = np.full(estimations.shape[1], np.nan, dtype=np.float64)
    np.divide(squared_deviations, counts - 1, out=variance, where=counts > 1)
    return mean, np.sqrt(variance)


def select_yearly_trial_estimation(trial_estimations, trial_index, positions):
    '''Select one trial/year without NumPy moving the time axis.

    The trial axis is removed before applying the integer-array time index.  This
    preserves the required ``(lat, lon, time)`` layout for NetCDF assignment.
    '''
    estimations = np.asarray(trial_estimations)
    if estimations.ndim != 4:
        raise ValueError(
            'Trial estimations must have shape (trial, lat, lon, time).'
        )
    if not 0 <= int(trial_index) < estimations.shape[0]:
        raise IndexError(f'Trial index is out of range: {trial_index}')
    selected = np.take(
        estimations[int(trial_index)],
        np.asarray(positions, dtype=np.int64),
        axis=-1,
    )
    return selected


def estimate_neural_aligned_batch(
    bundle,
    ssm_batch,
    static_scaled,
    static_raw,
    static_valid,
    time_values=None,
    target_start=0,
    target_stop=None,
    progress_callback=None,
    rainfall_batch=None,
    rainfall_dates=None,
):
    '''Estimate the seeded ensemble on every eligible finite-SSM target date.'''

    del static_raw
    n_pixels, n_time = ssm_batch.shape
    rain_scaler = bundle['precipitation_scaler']
    if rainfall_batch is None or rainfall_dates is None or time_values is None:
        raise ValueError('Event-rainfall models require daily rainfall and calendar dates.')
    if bundle.get('precipitation_intervals') != 1:
        raise ValueError('The published models use rainfall from one inter-event interval (N=1).')
    rainfall_batch = np.asarray(rainfall_batch)
    if rainfall_batch.shape != (n_pixels, len(rainfall_dates)):
        raise ValueError('Daily rainfall batch must match the pixel and rainfall-date axes.')
    models = bundle['models']

    estimation = np.full((n_pixels, n_time), np.nan, dtype=np.float32)
    estimation_std = np.full((n_pixels, n_time), np.nan, dtype=np.float32)
    trial_grid = np.full((len(models), n_pixels, n_time), np.nan, dtype=np.float32)
    valid_pixels = np.where(static_valid)[0]
    if len(valid_pixels) == 0:
        return estimation, estimation_std, trial_grid

    spatial_executor = bundle.get('spatial_executor')
    pixel_batch = int(bundle.get('pixel_batch', PIXEL_BATCH))
    prepared_time_days = prepare_observation_time_days(time_values, n_time)
    for batch_start in range(0, len(valid_pixels), pixel_batch):
        batch_idx = valid_pixels[batch_start:batch_start + pixel_batch]
        tasks = (
            (pixel_index, ssm_batch[pixel_index], prepared_time_days, target_start, target_stop)
            for pixel_index in batch_idx
        )
        mapper = map if spatial_executor is None else spatial_executor.map
        window_parts, owner_parts, target_parts = [], [], []
        for pixel_index, windows, target_indices in mapper(build_pixel_estimation_windows, tasks):
            if len(windows) == 0:
                continue
            intervals = event_interval_amounts(
                ssm_batch[pixel_index], time_values, rainfall_dates,
                rainfall_batch[pixel_index], lag_days=1,
            )
            rainfall = standardized_precipitation_windows(
                ssm_batch[pixel_index], target_indices, intervals, rain_scaler,
            )
            window_parts.append(np.concatenate((windows, rainfall[..., None]), axis=-1))
            owner_parts.append(np.full(len(windows), pixel_index, dtype=np.int64))
            target_parts.append(target_indices)

        if window_parts:
            owners = np.concatenate(owner_parts)
            targets = np.concatenate(target_parts)
            trial_estimations = ensemble_inference_sequence(
                models,
                np.concatenate(window_parts, axis=0).astype(np.float32),
                static_scaled[owners],
                bundle.get('trial_device_groups'),
                inference_batch=bundle.get('inference_batch', INFERENCE_BATCH),
            )
            mean, std = summarize_trial_estimations(trial_estimations)
            estimation[owners, targets] = mean.astype(np.float32)
            estimation_std[owners, targets] = std.astype(np.float32)
            for trial_index, trial_estimation in enumerate(trial_estimations):
                trial_grid[trial_index, owners, targets] = trial_estimation
        if progress_callback is not None:
            progress_callback(len(batch_idx))

    expected_support = observation_event_support_mask(
        ssm_batch,
        target_start=target_start,
        target_stop=target_stop,
        window_size=OBSERVATION_EVENT_SEQUENCE_LENGTH,
        prepared_time_days=prepared_time_days,
    ) & np.asarray(static_valid, dtype=bool)[:, None]
    supports = {'ensemble mean': estimation}
    if len(models) > 1:
        supports['ensemble spread'] = estimation_std
    for label, values in supports.items():
        actual = np.isfinite(values)
        if not np.array_equal(actual, expected_support):
            raise RuntimeError(
                f'Neural {label} support does not match eligible satellite support: '
                f'missing={int(np.count_nonzero(expected_support & ~actual))}, '
                f'unexpected={int(np.count_nonzero(actual & ~expected_support))}.'
            )
    return estimation, estimation_std, trial_grid


def ensemble_path(family):
    family = family.upper()
    if family not in static_profiles.FAMILY_DIRECTORIES:
        raise ValueError(f'Unsupported model family: {family}')
    return PATHS.results / static_profiles.FAMILY_DIRECTORIES[family] / 'Models' / f'{family}_ensemble.pt'


def load_ensemble(family, devices):
    '''Load the ten models from the pixel-trained SMAP ensemble checkpoint.'''

    family = family.upper()
    path = ensemble_path(family)
    if not path.is_file():
        raise FileNotFoundError(
            f'{path}: train the {family} ensemble first ({static_profiles.FAMILY_DIRECTORIES[family]}/01-04).'
        )
    payload = torch.load(path, map_location='cpu', weights_only=True)
    expected_architecture = 'full' if family == 'FNO' else 'LSTM'
    problems = [
        name for name, ok in (
            ('format_version', payload.get('format_version') == 1),
            ('architecture', payload.get('architecture') == expected_architecture),
            ('product', payload.get('product') == 'SMAP'),
            ('features', payload.get('features') == list(static_profiles.EVENT_FEATURES)),
            ('seeds', payload.get('seeds') == list(MODEL_SEEDS)),
            ('state_dicts', len(payload.get('state_dicts', ())) == len(MODEL_SEEDS)),
            ('development_dates', payload.get('development_dates') == neural_development_metadata()),
        ) if not ok
    ]
    if problems:
        raise ValueError(
            f'{path}: ensemble does not match the SMAP event-rainfall recipe ({", ".join(problems)}).'
        )
    scaler_df = pd.DataFrame(payload['static_scaler'])
    if scaler_df.Feature.tolist() != list(static_profiles.EVENT_FEATURES):
        raise ValueError(f'{path}: static scaler order differs from {list(static_profiles.EVENT_FEATURES)}')
    rain_scaler = {'mean': float(payload['rain_scaler']['mean']),
                   'std': float(payload['rain_scaler']['std'])}
    require_rainfall_file(payload['rain_scaler'].get('rainfall_file', ''), f'{path} (retrain 01-04)')

    params = payload['parameters']
    devices = list(devices)
    trial_devices = expand_trial_devices(len(MODEL_SEEDS), devices)
    models = []
    for state, device in zip(payload['state_dicts'], trial_devices):
        if family == 'FNO':
            model = FNOEvent(
                modes=int(params['modes']), width=int(params['width']),
                num_static_properties=len(static_profiles.EVENT_FEATURES), input_channels=4,
                dropout_static=float(params['dropout_static']),
                dropout_fc=float(params['dropout_fc']),
                include_precipitation=True,
            )
        else:
            model = LSTMEvent(
                hidden_size=int(params['hidden_size']), num_layers=int(params['num_layers']),
                num_static_properties=len(static_profiles.EVENT_FEATURES), input_channels=3,
                dropout_static=float(params['dropout_static']),
                dropout_fc=float(params['dropout_fc']),
                include_precipitation=True,
            )
        model = model.to(device)
        model.load_state_dict(state)
        model.eval()
        models.append(model)

    return {
        'architecture': family,
        'models': models,
        'model_seeds': list(MODEL_SEEDS),
        'trial_count': len(models),
        'trial_device_groups': create_trial_parallel_backend(models, devices),
        'scaler_df': scaler_df,
        'dynamic_channels': list(EVENT_PRECIPITATION_CHANNELS),
        'precipitation_scaler': rain_scaler,
        'precipitation_intervals': 1,
        'rainfall_variable': RAINFALL_VARIABLE,
        'metadata': {
            **neural_development_metadata(),
            'static_schema_version': static_profiles.EVENT_SCHEMA,
            'static_features': ','.join(static_profiles.EVENT_FEATURES),
            'precipitation_source': 'daily rainfall; N=1; previous SMAP event through previous day',
            'precipitation_file': RAINFALL_FILE,
        },
    }


def prepare_bundles(models, data, gpus, fno_architecture, lstm_architecture, workers=DEFAULT_WORKERS):
    '''Load saved neural ensembles.'''
    del data, fno_architecture, lstm_architecture
    workers = int(workers)
    if workers < 1:
        raise ValueError('workers must be at least 1.')
    neural_models = [name for name in models if name in NEURAL_MODEL_CHOICES]
    if not neural_models:
        return {}
    if ACTIVE_STATIC_PROFILE != static_profiles.EVENT_PROFILE:
        raise ValueError('Published neural estimation requires the NDVI/bulk-density event-rainfall recipe.')
    devices = resolve_estimation_devices(gpus)
    pixel_batch, inference_batch = resolve_estimation_batch_sizes(devices)
    bundles = {}
    for model_name in neural_models:
        bundle = load_ensemble(neural_model_family(model_name), devices)
        bundle['pixel_batch'] = pixel_batch
        bundle['inference_batch'] = inference_batch
        bundle['spatial_executor'] = ThreadPoolExecutor(max_workers=workers) if workers > 1 else None
        bundles[model_name] = bundle
    return bundles


def close_trial_parallel_backends(bundles):
    for bundle in bundles.values():
        executors = {bundle.get('spatial_executor')}
        for group in bundle.get('trial_device_groups') or ():
            executors.add(group.get('trial_executor'))
        for executor in executors:
            if executor is not None:
                executor.shutdown(wait=True)


def _estimation_variable(ds, name, long_name):
    variable = ds.createVariable(
        name, 'f4', ('lat', 'lon', 'time'), zlib=True, complevel=4, fill_value=np.float32(np.nan)
    )
    variable.units = 'cm3/cm3'
    variable.long_name = long_name
    return variable


def create_yearly_files(models, data, lat_values, lon_values, selected_time, year_positions, time_attrs, bundles):
    '''Create empty yearly files (``run_complete = 0``) for every model.'''
    for model_name in models:
        os.makedirs(os.path.join(model_output_dir(model_name), 'Yearly'), exist_ok=True)
        for year, positions in year_positions.items():
            with nc.Dataset(yearly_output_path(model_name, data, year), 'w', format='NETCDF4') as ds:
                ds.createDimension('lat', len(lat_values))
                ds.createDimension('lon', len(lon_values))
                ds.createDimension('time', len(positions))
                lat_var = ds.createVariable('lat', 'f4', ('lat',))
                lon_var = ds.createVariable('lon', 'f4', ('lon',))
                time_var = ds.createVariable('time', selected_time.dtype, ('time',))
                lat_var[:] = lat_values
                lon_var[:] = lon_values
                time_var[:] = selected_time[positions]
                lat_var.units = 'degrees_north'
                lon_var.units = 'degrees_east'
                for attr_name, attr_value in time_attrs.items():
                    time_var.setncattr(attr_name, attr_value)

                if model_name == 'EF':
                    _estimation_variable(ds, 'RZSM', f'EF {data} root-zone soil moisture estimation')
                    ds.EF_filter_T_days = ACTIVE_EF_T
                    ds.EF_filter_policy = EF_POLICY
                    ds.EF_spin_up_days = EF_SPIN_UP_DAYS
                    ds.output_support = f'finite {data} SSM dates after the EF spin-up'
                else:
                    bundle = bundles[model_name]
                    trial_count = bundle['trial_count']
                    for trial_index, seed in enumerate(bundle['model_seeds'], start=1):
                        trial_var = _estimation_variable(
                            ds, f'RZSM_Trial_{trial_index}',
                            f'{model_name} {data} trial {trial_index} RZSM estimation',
                        )
                        trial_var.seed = seed
                    _estimation_variable(
                        ds, 'RZSM_avg',
                        f'{model_name} {data} mean RZSM estimation across {trial_count} seeded trials',
                    )
                    std_var = _estimation_variable(
                        ds, 'RZSM_std',
                        f'{model_name} {data} sample standard deviation across {trial_count} seeded trials',
                    )
                    std_var.ddof = 1
                    ds.model_seeds = ','.join(str(seed) for seed in bundle['model_seeds'])
                    ds.trial_count = trial_count
                    for name, value in bundle['metadata'].items():
                        ds.setncattr(name, value)
                    ds.neural_recipe = NEURAL_RECIPE_EVENT
                    ds.dynamic_channels = ','.join(bundle['dynamic_channels'])
                    ds.sequence_length_observations = int(OBSERVATION_EVENT_SEQUENCE_LENGTH)
                    ds.gap_scale_days = float(OBSERVATION_EVENT_GAP_SCALE_DAYS)
                    ds.precipitation_variable = bundle['rainfall_variable']
                    ds.precipitation_lag_days = 1
                    ds.precipitation_transform = 'log1p(interval_mm), standardized with the training scaler'
                    ds.output_support = (
                        f'finite {data} SSM dates with at least '
                        f'{OBSERVATION_EVENT_SEQUENCE_LENGTH} valid observations'
                    )
                ds.estimation_model = model_name
                ds.source_data = data
                ds.estimation_start_date = str(ESTIMATION_START_DATE)
                ds.estimation_end_date = str(COMMON_END_DATE)
                ds.run_complete = 0


def write_yearly_chunk(model_name, data, lat_offset, estimation, year_positions, estimation_std=None, trial_estimations=None):
    '''Write one latitude chunk ``(lat, lon, time)`` into every yearly file.'''
    if trial_estimations is not None:
        trial_estimations = np.asarray(trial_estimations, dtype=np.float32)
        if trial_estimations.shape[1:] != estimation.shape:
            raise ValueError(
                f'Trial estimation shape {trial_estimations.shape[1:]} differs from '
                f'the ensemble mean {estimation.shape}.'
            )
    destination = np.s_[lat_offset:lat_offset + estimation.shape[0], :, :]
    for year, positions in year_positions.items():
        with nc.Dataset(yearly_output_path(model_name, data, year), 'a') as ds:
            if trial_estimations is None:
                ds.variables['RZSM'][destination] = estimation[:, :, positions]
                continue
            if trial_estimations.shape[0] != int(ds.trial_count):
                raise ValueError(
                    f'{model_name} {data} {year}: expected {ds.trial_count} trials, '
                    f'received {trial_estimations.shape[0]}.'
                )
            ds.variables['RZSM_avg'][destination] = estimation[:, :, positions]
            for trial_index in range(trial_estimations.shape[0]):
                ds.variables[f'RZSM_Trial_{trial_index + 1}'][destination] = (
                    select_yearly_trial_estimation(trial_estimations, trial_index, positions)
                )
            if estimation_std is not None:
                ds.variables['RZSM_std'][destination] = estimation_std[:, :, positions]


def mark_yearly_files_complete(models, data, year_positions):
    for model_name in models:
        for year in year_positions:
            with nc.Dataset(yearly_output_path(model_name, data, year), 'a') as ds:
                ds.run_complete = 1


def count_progress_pixels(models, bundles, static_ds, lat_slice, lon_slice):
    '''Number of CONUS pixels each model processes (progress-bar totals).'''
    conus_mask = np.asarray(static_ds.variables['CONUS_mask'][lat_slice, lon_slice]).astype(bool).reshape(-1)
    totals = {}
    for model_name in models:
        if model_name == 'EF':
            totals[model_name] = int(np.count_nonzero(conus_mask))
        else:
            _, _, static_valid, _ = read_static_chunk(
                static_ds, lat_slice, lon_slice, bundles[model_name]['scaler_df']
            )
            totals[model_name] = int(np.count_nonzero(static_valid & conus_mask))
    return totals


def process_conus(data, models, gpus=(), workers=DEFAULT_WORKERS, *, ef_filter=None):
    '''Estimate and write yearly CONUS root-zone soil moisture.'''
    global ACTIVE_EF_T
    models = [model for model in models if model in MODEL_SAVE_MAP]
    if not models:
        return
    if 'EF' in models:
        ACTIVE_EF_T = load_ef_t(None, data, required=True)

    input_path = product_path(data)
    static_path = static_input_path()
    if not os.path.exists(input_path):
        raise FileNotFoundError(f'Missing {data} input product: {input_path}')
    if not os.path.exists(static_path):
        raise FileNotFoundError(f'Missing model input grid: {static_path}. Run 01_Model_input.py first.')
    if any(model in NEURAL_MODEL_CHOICES for model in models) and not os.path.exists(rainfall_input_path()):
        raise FileNotFoundError(
            f"{rainfall_input_path()}: daily GPM rainfall (time, lat, lon) with "
            "'daily_precipitation_sum' is required."
        )

    bundles = prepare_bundles(models, data, gpus, 'FNO', 'LSTM', workers=workers)
    try:
        with nc.Dataset(input_path, 'r') as product_ds, nc.Dataset(static_path, 'r') as static_ds:
            static_profiles.validate_grid_schema(static_ds, ACTIVE_STATIC_PROFILE, static_path)
            if 'CONUS_mask' not in static_ds.variables:
                raise KeyError(f'{static_path} is missing CONUS_mask. Rerun 01_Model_input.py.')
            product_lat_name = coord_name(product_ds, ('lat', 'latitude'))
            product_lon_name = coord_name(product_ds, ('lon', 'longitude'))
            product_lat = coord_axis_1d(product_ds, product_lat_name, 'lat')
            product_lon = coord_axis_1d(product_ds, product_lon_name, 'lon')
            static_lat = coord_axis_1d(static_ds, coord_name(static_ds, ('lat', 'latitude')), 'lat')
            static_lon = coord_axis_1d(static_ds, coord_name(static_ds, ('lon', 'longitude')), 'lon')

            lat_slice, lon_slice = conus_slices(static_lat, static_lon)
            product_lat_slice, product_lon_slice = conus_slices(product_lat, product_lon)
            lat_values = static_lat[lat_slice]
            lon_values = static_lon[lon_slice]
            flip_lat = axis_flip_to_reference(lat_values, product_lat[product_lat_slice], f'{data} latitude')
            flip_lon = axis_flip_to_reference(lon_values, product_lon[product_lon_slice], f'{data} longitude')

            selected_idx, selected_time, selected_dates, time_days, year_positions, time_attrs = (
                selected_time_info(product_ds, YEARS)
            )
            create_yearly_files(models, data, lat_values, lon_values, selected_time, year_positions, time_attrs, bundles)
            target_start = int(np.searchsorted(selected_dates, ESTIMATION_START_DATE, side='left'))

            totals = count_progress_pixels(models, bundles, static_ds, lat_slice, lon_slice)
            progress = {
                model: tqdm(total=totals[model], desc=f'{model} | {data}', unit='pixel', leave=False)
                for model in models
            }
            n_lat = len(lat_values)
            for local_start in range(0, n_lat, CHUNK_LAT):
                local_end = min(local_start + CHUNK_LAT, n_lat)
                static_chunk = slice(lat_slice.start + local_start, lat_slice.start + local_end)
                if flip_lat:
                    product_start, product_end = n_lat - local_end, n_lat - local_start
                else:
                    product_start, product_end = local_start, local_end
                product_chunk = slice(
                    product_lat_slice.start + product_start, product_lat_slice.start + product_end
                )

                ssm_grid = read_product_chunk(
                    product_ds, PRODUCTS[data]['variable'], product_chunk, product_lon_slice,
                    selected_idx, product_lat_name, product_lon_name,
                )
                if flip_lat:
                    ssm_grid = ssm_grid[::-1, :, :]
                if flip_lon:
                    ssm_grid = ssm_grid[:, ::-1, :]
                ssm_flat = clean_ssm(ssm_grid, data).reshape(-1, ssm_grid.shape[-1])
                conus_mask = np.asarray(
                    static_ds.variables['CONUS_mask'][static_chunk, lon_slice]
                ).astype(bool).reshape(-1)
                ssm_flat[~conus_mask, :] = np.nan

                if 'EF' in models:
                    ef_flat = (ef_filter or exponential_filter_batch)(ssm_flat, time_days, T=ACTIVE_EF_T)
                    write_yearly_chunk('EF', data, local_start, ef_flat.reshape(ssm_grid.shape), year_positions)
                    progress['EF'].update(int(np.count_nonzero(conus_mask)))

                if bundles:
                    yy, xx = np.meshgrid(static_lat[static_chunk], static_lon[lon_slice], indexing='ij')
                    variables = {bundle['rainfall_variable'] for bundle in bundles.values()}
                    if len(variables) != 1:
                        raise ValueError('Neural families must use the same daily rainfall variable.')
                    rain_dates, rain_flat, _ = extract_daily_pixels(
                        rainfall_input_path(), variables.pop(),
                        pd.DataFrame({'lat': yy.ravel(), 'lon': xx.ravel()}),
                        selected_dates[0], selected_dates[-1] - np.timedelta64(1, 'D'),
                        report_progress=False,
                    )

                for model_name, bundle in bundles.items():
                    _, static_scaled, static_valid, _ = read_static_chunk(
                        static_ds, static_chunk, lon_slice, bundle['scaler_df']
                    )
                    static_valid &= conus_mask
                    mean, std, trials = estimate_neural_aligned_batch(
                        bundle, ssm_flat, static_scaled, None, static_valid,
                        time_values=selected_dates,
                        target_start=target_start,
                        target_stop=len(selected_dates),
                        progress_callback=progress[model_name].update,
                        rainfall_batch=rain_flat,
                        rainfall_dates=rain_dates,
                    )
                    write_yearly_chunk(
                        model_name, data, local_start,
                        mean.reshape(ssm_grid.shape),
                        year_positions,
                        estimation_std=std.reshape(ssm_grid.shape),
                        trial_estimations=trials.reshape((bundle['trial_count'],) + ssm_grid.shape),
                    )

            for bar in progress.values():
                bar.close()
            mark_yearly_files_complete(models, data, year_positions)
    finally:
        close_trial_parallel_backends(bundles)


def merged_output_path(model_name, data):
    return os.path.join(model_output_dir(model_name), f'{model_name}_{data}_estimation.nc')


def copy_variable_attrs(source_var, target_var):
    for attr_name in source_var.ncattrs():
        if attr_name != '_FillValue':
            target_var.setncattr(attr_name, source_var.getncattr(attr_name))


def estimation_variable_names(model_name, trial_count):
    if model_name not in NEURAL_MODEL_CHOICES:
        return ['RZSM']
    return [f'RZSM_Trial_{index}' for index in range(1, trial_count + 1)] + ['RZSM_avg', 'RZSM_std']


def assert_matching_grid(reference_ds, target_ds, target_path):
    '''Reject a yearly file whose grid or model settings differ from the first year.'''
    for coord_name in ('lat', 'lon'):
        ref_values = np.asarray(reference_ds.variables[coord_name][:])
        target_values = np.asarray(target_ds.variables[coord_name][:])
        if ref_values.shape != target_values.shape:
            raise ValueError(
                f'{coord_name} shape mismatch in {target_path}: {target_values.shape} vs {ref_values.shape}'
            )
        max_diff = np.nanmax(np.abs(ref_values - target_values))
        if max_diff > 1e-5:
            raise ValueError(f'{coord_name} coordinate mismatch in {target_path}: max diff={max_diff}')
    for attr_name in CONSISTENT_ATTRIBUTES:
        reference_value = getattr(reference_ds, attr_name, None)
        target_value = getattr(target_ds, attr_name, None)
        if reference_value != target_value:
            raise ValueError(
                f'{attr_name} mismatch in {target_path}: {target_value!r} vs {reference_value!r}'
            )


def collect_yearly_files(model_name, data, years):
    files = [
        (int(year), os.fspath(existing_result_path(yearly_output_path(model_name, data, int(year)))))
        for year in years
    ]
    missing = [(year, path) for year, path in files if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(
            f'Missing yearly estimation files for {model_name} {data}: '
            f'{", ".join(str(year) for year, _ in missing)}. First missing path: {missing[0][1]}'
        )
    return files


def create_merged_dataset(output_path, first_ds, model_name, data, total_time):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output_ds = nc.Dataset(output_path, 'w', format='NETCDF4')
    output_ds.createDimension('lat', len(first_ds.dimensions['lat']))
    output_ds.createDimension('lon', len(first_ds.dimensions['lon']))
    output_ds.createDimension('time', total_time)

    for coord_name in ('lat', 'lon'):
        source = first_ds.variables[coord_name]
        target = output_ds.createVariable(coord_name, source.dtype, (coord_name,))
        target[:] = source[:]
        copy_variable_attrs(source, target)
    source_time = first_ds.variables['time']
    target_time = output_ds.createVariable('time', source_time.dtype, ('time',))
    copy_variable_attrs(source_time, target_time)

    for variable_name in estimation_variable_names(model_name, int(getattr(first_ds, 'trial_count', 0))):
        source_variable = first_ds.variables[variable_name]
        target_variable = output_ds.createVariable(
            variable_name,
            source_variable.dtype,
            source_variable.dimensions,
            zlib=True,
            complevel=4,
            fill_value=np.float32(np.nan),
        )
        copy_variable_attrs(source_variable, target_variable)

    for attr_name in first_ds.ncattrs():
        output_ds.setncattr(attr_name, first_ds.getncattr(attr_name))
    output_ds.estimation_model = model_name
    output_ds.source_data = data
    output_ds.run_complete = 0
    return output_ds


def merge_one(model_name, data, years):
    '''Merge the yearly files of one model/product, writing via a temporary file.'''
    yearly_files = collect_yearly_files(model_name, data, years)
    total_time = 0
    for _, path in yearly_files:
        with nc.Dataset(path, 'r') as ds:
            if int(getattr(ds, 'run_complete', 0)) != 1:
                raise RuntimeError(f'Yearly estimation is incomplete: {path}')
            trial_count = int(getattr(ds, 'trial_count', 0))
            if model_name in NEURAL_MODEL_CHOICES and trial_count < 1:
                raise RuntimeError(f'Neural yearly estimation has no trial_count: {path}')
            missing = [
                name for name in estimation_variable_names(model_name, trial_count)
                if name not in ds.variables
            ]
            if missing:
                raise RuntimeError(f'Yearly estimation is missing {", ".join(missing)}: {path}')
            total_time += len(ds.dimensions['time'])

    output_path = merged_output_path(model_name, data)
    tmp_output_path = f'{output_path}.tmp.{os.getpid()}'
    with nc.Dataset(yearly_files[0][1], 'r') as first_ds:
        output_ds = create_merged_dataset(tmp_output_path, first_ds, model_name, data, total_time)
    try:
        try:
            n_lat = len(output_ds.dimensions['lat'])
            variable_names = estimation_variable_names(model_name, int(getattr(output_ds, 'trial_count', 0)))
            time_offset = 0
            for _, path in tqdm(yearly_files, desc=f'Merging {model_name} {data}', leave=False):
                with nc.Dataset(path, 'r') as source_ds:
                    assert_matching_grid(output_ds, source_ds, path)
                    n_time = len(source_ds.dimensions['time'])
                    output_ds.variables['time'][time_offset:time_offset + n_time] = source_ds.variables['time'][:]
                    for lat_start in range(0, n_lat, MERGE_CHUNK_LAT):
                        lat_end = min(lat_start + MERGE_CHUNK_LAT, n_lat)
                        for variable_name in variable_names:
                            output_ds.variables[variable_name][
                                lat_start:lat_end, :, time_offset:time_offset + n_time
                            ] = source_ds.variables[variable_name][lat_start:lat_end, :, :]
                    time_offset += n_time
            merged_time = np.asarray(output_ds.variables['time'][:], dtype=float)
            if len(merged_time) > 1 and not np.all(np.diff(merged_time) > 0):
                raise ValueError(f'Merged time axis is not strictly increasing for {model_name} {data}.')
            output_ds.run_complete = 1
        finally:
            output_ds.close()
        os.replace(tmp_output_path, output_path)
    finally:
        if os.path.exists(tmp_output_path):
            os.remove(tmp_output_path)


def merge_patches(args):
    '''Merge yearly files for each selected model and satellite product.'''
    years = getattr(args, 'years', ESTIMATION_YEARS)
    for model_name in args.model:
        if model_name not in MODEL_SAVE_MAP:
            raise ValueError(f'Unsupported model: {model_name}')
        for data in args.data:
            merge_one(model_name, data, years)


def parse_merge_args(argv=None):
    parser = argparse.ArgumentParser(description='Merge yearly CONUS RZSM estimation files.')
    parser.add_argument(
        '--model', nargs='+', default=[DEFAULT_FNO_MODEL, DEFAULT_LSTM_MODEL],
        choices=['EF', *NEURAL_MODEL_CHOICES],
    )
    parser.add_argument(
        '--data', nargs='+', default=['SMAP'], choices=['ASCAT', 'SMAP'],
        help='Estimation source to merge (default: SMAP). ASCAT has EF estimations only.',
    )
    parser.add_argument('--years', nargs='+', type=int, default=ESTIMATION_YEARS)
    return parser.parse_args(argv)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, epilog='Use the merge command to merge existing yearly files.')
    parser.add_argument(
        '--data', nargs='+', choices=('SMAP', 'ASCAT'), default=['SMAP', 'ASCAT'],
        help='Satellite product(s) to process (default: SMAP ASCAT).',
    )
    return parser.parse_args(argv)


def apply_spin_up_mask(values, time_days, spin_up_days=EF_SPIN_UP_DAYS):
    time_days = np.asarray(time_days, dtype=float)
    if values.shape[-1] != time_days.size:
        raise ValueError(
            f'Spin-up time axis mismatch: data has {values.shape[-1]} samples, '
            f'time has {time_days.size}.'
        )
    if time_days.size:
        values[..., (time_days - time_days[0]) < spin_up_days] = np.nan
    return values


def exponential_filter_batch(ssm, time_days, T=None, mask_spin_up=True):
    '''Run the recursive EF over ``(pixel, time)`` SSM and emit values on valid SSM dates.'''
    if T is None:
        T = ACTIVE_EF_T
    if T is None:
        raise ValueError('No EF timescale is set; process_conus loads it from the calibration.')

    n_pixels, n_time = ssm.shape
    if time_days.size != n_time:
        raise ValueError(
            f'EF time axis mismatch: SSM has {n_time} samples, time has {time_days.size}.'
        )
    if T <= 0:
        raise ValueError('EF T must be positive.')

    rzsm = np.full((n_pixels, n_time), np.nan, dtype=np.float32)
    current = np.full(n_pixels, np.nan, dtype=np.float32)
    K_prev = np.ones(n_pixels, dtype=np.float32)
    last_valid_time = np.zeros(n_pixels, dtype=np.float32)
    initialized = np.zeros(n_pixels, dtype=bool)

    for t in range(n_time):
        valid = ~np.isnan(ssm[:, t])
        new_init = valid & ~initialized
        if np.any(new_init):
            current[new_init] = ssm[new_init, t]
            K_prev[new_init] = 1.0
            last_valid_time[new_init] = time_days[t]
            initialized[new_init] = True

        update = valid & initialized & ~new_init
        if np.any(update):
            delta_t = time_days[t] - last_valid_time[update]
            if np.any(delta_t <= 0):
                raise ValueError('Valid SSM observation times must be strictly increasing.')
            decay = np.exp(-delta_t / T).astype(np.float32)
            K_n = K_prev[update] / (K_prev[update] + decay)
            current[update] = current[update] + K_n * (ssm[update, t] - current[update])
            K_prev[update] = K_n
            last_valid_time[update] = time_days[t]

        rzsm[valid, t] = current[valid]

    if mask_spin_up:
        return apply_spin_up_mask(rzsm, time_days)
    return rzsm


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ['merge']:
        merge_patches(parse_merge_args(argv[1:]))
        return

    args = parse_args(argv)

    print(f'Writing EF estimates under {estimation_output_root(DATA_ROOT)}')
    for data in args.data:
        print(f'EF {data}: T={load_ef_t(DATA_ROOT, data):g} days')
        process_conus(data, ['EF'], workers=10)

    for data in args.data:
        merge_one('EF', data, ESTIMATION_YEARS)


if __name__ == '__main__':
    main()
