"""Build the FNO input tables, static scaler and rainfall intervals."""

import argparse
from importlib import import_module
from pathlib import Path
import sys

import netCDF4 as nc
import numpy as np
import pandas as pd
from tqdm import tqdm

publication_root = Path(__file__).resolve().parents[1]
if str(publication_root) not in sys.path:
    sys.path.insert(0, str(publication_root))
workflow = import_module('Fourier_Neural_Operator.04_FNO_train').workflow

from core import pixel_training as training, static_input_profiles as profiles
from core.pixel_training import SSM_VARIABLE, TARGET_VARIABLE, fit_feature_scaler
from core.pixel_workflow import (
    AREAS,
    FEATURES,
    METADATA_COLUMNS,
    RAIN_LAG_DAYS,
    pixel_rainfall_file,
    read_pixel_precipitation,
)
from core.precipitation_inputs import (
    RAINFALL_FILE,
    event_interval_amounts,
    event_windows,
    precipitation_windows,
    require_rainfall_file,
)
from core.project_paths import DATA_ROOT
from core.rzsm_pipeline_config import build_observation_event_sequences, split_boundaries


MIN_VALID_OBSERVATIONS = 100


def _finite_scalar(variable, lat_idx, lon_idx):
    value = variable[lat_idx, lon_idx]
    if np.ma.is_masked(value):
        return np.nan
    value = float(value)
    return value if np.isfinite(value) else np.nan


def collect_csv_inputs(workflow, area, data_root):
    """Pixels of one area with finite static features and enough SSM/RZSM days."""
    if area not in AREAS:
        raise ValueError(f'Unknown area: {area}')
    profile = profiles.DEFAULT_PROFILE
    features = profiles.features(profile)
    columns = [*METADATA_COLUMNS, *features]
    ismn = workflow.ismn
    pixel_list_path = ismn / 'Station_TCA_Screening' / f'Common_{area}_pixel_list.csv'
    if not pixel_list_path.exists():
        raise FileNotFoundError(f'Required {area} pixel list is missing: {pixel_list_path}')
    try:
        pixel_list = pd.read_csv(pixel_list_path)
    except pd.errors.EmptyDataError:
        pixel_list = pd.DataFrame(columns=['lat_idx', 'lon_idx'])
    if not {'lat_idx', 'lon_idx'} <= set(pixel_list.columns):
        raise ValueError(f'{pixel_list_path}: missing lat_idx/lon_idx columns')
    pixel_list = pixel_list.drop_duplicates(['lat_idx', 'lon_idx'])
    if pixel_list.empty:
        raise ValueError(f'Required {area} pixel list is empty: {pixel_list_path}')

    grid_path = str(profiles.grid_path(data_root, profile, read_existing=True))
    rows = []
    with nc.Dataset(grid_path) as grid:
        profiles.validate_grid_schema(grid, profile, grid_path)
        lats, lons = grid['lat'][:], grid['lon'][:]
        for row in tqdm(pixel_list.itertuples(), total=len(pixel_list), desc=f'Extracting {area}', leave=False):
            lat_idx, lon_idx = int(row.lat_idx), int(row.lon_idx)
            static = {name: _finite_scalar(grid[name], lat_idx, lon_idx) for name in features}
            pixel_path = ismn / f'ISMN_{area}' / f'{lat_idx}_{lon_idx}.nc'
            if not all(np.isfinite(v) for v in static.values()) or not pixel_path.exists():
                continue
            with nc.Dataset(pixel_path) as pixel:
                missing = [n for n in (TARGET_VARIABLE, SSM_VARIABLE, 'Precipitation') if n not in pixel.variables]
                if missing:
                    raise KeyError(f'{pixel_path} is missing variables: {", ".join(missing)}')
                ssm = np.isfinite(pixel[SSM_VARIABLE][:].filled(np.nan))
                valid_ssm = int(ssm.sum())
                valid_rzsm = int((ssm & np.isfinite(pixel[TARGET_VARIABLE][:].filled(np.nan))).sum())
            if min(valid_ssm, valid_rzsm) < MIN_VALID_OBSERVATIONS:
                continue
            rows.append({'lat_idx': lat_idx, 'lon_idx': lon_idx, 'lat': float(lats[lat_idx]),
                         'lon': float(lons[lon_idx]), 'valid_SSM': valid_ssm, 'valid_RZSM': valid_rzsm, **static})
    frame = pd.DataFrame(rows, columns=columns)
    if frame.empty:
        raise RuntimeError(f'No usable {area} pixels among {len(pixel_list)} listed pixels')
    return frame


def generate_csv_inputs(workflow, area, data_root):
    """Write the area table; on failure remove any older table so it cannot be reused."""
    output = workflow.input_csv(area)
    try:
        frame = collect_csv_inputs(workflow, area, data_root)
        output.parent.mkdir(parents=True, exist_ok=True)
        temp = output.with_name(output.name + '.tmp')
        frame.to_csv(temp, index=False)
        temp.replace(output)
    except BaseException:
        output.unlink(missing_ok=True)
        raise
    return frame


def prepare_rainfall(workflow, table):
    """Save rainfall between SMAP observations and fit its scaler on training windows.

    The interval ending at observation b holds the daily totals (06:00-06:00
    local days) from the previous observation a through day b-1; rain on
    day b is excluded. Only dates before the test period are read.
    """
    destination = workflow.inputs / 'Rain'
    destination.mkdir(parents=True, exist_ok=True)
    train_pixels = training.spatial_training_pixels(len(table))
    coverage, training_values = [], []
    for position, row in enumerate(table.itertuples()):
        pixel = f'{int(row.lat_idx)}_{int(row.lon_idx)}'
        with nc.Dataset(workflow.ismn / 'ISMN_Train' / f'{pixel}.nc') as ds:
            ssm, y, dates, start, _ = training.read_pixel_training_series(ds)
            daily = read_pixel_precipitation(ds, dates)
            require_rainfall_file(pixel_rainfall_file(ds), f'{pixel} pixel file')
        train_stop, val_stop = split_boundaries(dates)
        ssm, y, dates, daily = ssm[:val_stop], y[:val_stop], dates[:val_stop], daily[:val_stop]
        y = training.mask_to_ssm_observations(y, ssm)
        intervals = event_interval_amounts(ssm, dates, dates, daily, RAIN_LAG_DAYS)
        seen = position in train_pixels
        partitions = ({'train': (start, train_stop), 'temporal_validation': (train_stop, val_stop)}
                      if seen else {'spatial_validation': (start, val_stop)})
        record = {'pixel': pixel, 'spatial_partition': 'training' if seen else 'validation',
                  'train_windows': 0, 'temporal_validation_windows': 0, 'spatial_validation_windows': 0}
        for split, (lo, hi) in partitions.items():
            _, _, indices = build_observation_event_sequences(ssm, y, time_values=dates,
                                                              target_start=lo, target_stop=hi)
            try:
                precipitation_windows(ssm, indices, intervals)
            except ValueError as error:
                raise ValueError(f'{pixel}, {split}: {error}') from error
            record[split + '_windows'] = len(indices)
            if split == 'train' and len(indices):

                used = np.unique(event_windows(ssm, indices)[:, 1:])
                training_values.append(np.log1p(intervals[used].astype(np.float64)))
        coverage.append(record)
        np.savez_compressed(destination / f'{pixel}.npz', dates=dates, interval_mm=intervals)
    if not training_values:
        raise ValueError('No training precipitation intervals')
    values = np.concatenate(training_values)
    scale = float(values.std(ddof=0))
    scaler = {'mean': float(values.mean()), 'std': scale if scale > 1e-8 else 1.,
              'transform': 'log1p(interval_mm)', 'training_interval_count': len(values),

              'rainfall_file': RAINFALL_FILE}
    pd.DataFrame([scaler]).to_csv(destination / 'Scaler.csv', index=False)
    pd.DataFrame(coverage).to_csv(destination / 'Coverage.csv', index=False)
    return scaler


def build_argument_parser():
    return argparse.ArgumentParser(description=__doc__, allow_abbrev=False)


if __name__ == '__main__':
    build_argument_parser().parse_args()
    for area in AREAS:
        generate_csv_inputs(workflow, area, DATA_ROOT)
    training_pixels = workflow.training_table()
    fit_feature_scaler(training_pixels, FEATURES).to_csv(workflow.inputs / 'Static_scaler.csv', index=False)
    prepare_rainfall(workflow, training_pixels)
    print("01_FNO_input_csv.py is finished.")
