"""Estimate RZSM at the independent test pixels with the trained LSTM ensemble."""

import argparse
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from importlib import import_module
from pathlib import Path
import sys

import netCDF4 as nc
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

publication_root = Path(__file__).resolve().parents[1]
if str(publication_root) not in sys.path:
    sys.path.insert(0, str(publication_root))
workflow = import_module('Long_Short-Term_Memory.04_LSTM_train').workflow

from core import pixel_training as training
from core.pixel_workflow import (
    FEATURES,
    ESTIMATION_BATCH,
    RAIN_LAG_DAYS,
    pixel_rainfall_file,
    read_pixel_precipitation,
)
from core.precipitation_inputs import (
    event_interval_amounts,
    require_rainfall_file,
    standardized_precipitation_windows,
)
from core.rzsm_pipeline_config import (
    build_observation_event_sequences,
    dates_from_variable,
    pixel_context_period_indices,
)


_ESTIMATION = None


def initialize_estimation(workflow):
    global _ESTIMATION
    torch.set_num_threads(1)
    _ESTIMATION = workflow.load_ensemble()


def estimate_pixel(workflow, task):
    area, row = task
    bundle, models = _ESTIMATION
    pixel = f"{int(row['lat_idx'])}_{int(row['lon_idx'])}"
    with nc.Dataset(workflow.ismn / f'ISMN_{area}' / f'{pixel}.nc') as ds:
        ssm, obs, dates, start, stop = training.read_pixel_training_series(ds)
        daily_rain = read_pixel_precipitation(ds, dates)
        require_rainfall_file(pixel_rainfall_file(ds), f'{area}/{pixel} pixel file')
        indices, _, _ = pixel_context_period_indices(dates_from_variable(ds['time']), require_complete=True)
        raw_time = np.asarray(ds['time'][:])[indices]
        units, calendar = ds['time'].units, getattr(ds['time'], 'calendar', 'standard')
        ef = (np.ma.filled(ds['SMAP_RZSM'][:], np.nan).reshape(-1)[indices]
              if 'SMAP_RZSM' in ds.variables else np.full(len(dates), np.nan))
    windows, _, targets = build_observation_event_sequences(ssm, time_values=dates, target_start=start, target_stop=stop)
    if not len(targets):
        return pixel, False
    require_rainfall_file(bundle['rain_scaler'].get('rainfall_file', ''), 'ensemble rain scaler (retrain 01-04)')
    intervals = event_interval_amounts(ssm, dates, dates, daily_rain, RAIN_LAG_DAYS)
    rain = standardized_precipitation_windows(ssm, targets, intervals, bundle['rain_scaler'])
    dynamic = np.concatenate([windows, rain[..., None]], axis=-1)
    frame = pd.DataFrame([row])
    training.apply_feature_scaler(frame, pd.DataFrame(bundle['static_scaler']))
    static = torch.tensor(frame[FEATURES].to_numpy(), dtype=torch.float32)
    estimations = np.full((len(models), len(dates)), np.nan, dtype=np.float32)
    with torch.inference_mode():
        for i, model in enumerate(models):
            for offset in range(0, len(targets), ESTIMATION_BATCH):
                x = torch.tensor(dynamic[offset:offset + ESTIMATION_BATCH], dtype=torch.float32)
                estimations[i, targets[offset:offset + len(x)]] = model(x, static.expand(len(x), -1)).numpy().reshape(-1)
    mean = np.full(len(dates), np.nan, dtype=np.float32)
    spread = mean.copy()
    mean[targets] = estimations[:, targets].mean(axis=0)
    spread[targets] = estimations[:, targets].std(axis=0, ddof=1)
    output = workflow.estimation_directory(area) / f'{pixel}_estimation.nc'
    temp = output.with_name(output.name + f'.tmp.{os.getpid()}')
    try:
        with nc.Dataset(temp, 'w') as ds:
            ds.createDimension('time', len(dates))
            t = ds.createVariable('time', 'i4', ('time',))
            t[:] = raw_time
            t.units, t.calendar = units, calendar
            values = dict(SMAP_SSM=ssm, RZSM=obs, EF_RZSM=ef, RZSM_avg=mean, RZSM_std=spread)
            values.update({f'RZSM_trial_{i + 1}': p for i, p in enumerate(estimations)})
            for name, value in values.items():
                v = ds.createVariable(name, 'f4', ('time',))
                v[:] = value
                v.units = 'm3 m-3'
                if name.startswith('RZSM_trial_'):
                    v.seed = bundle['seeds'][int(name.rsplit('_', 1)[1]) - 1]
            ds['RZSM_std'].ddof = 1
            ds.trial_count = len(models)
            ds.static_features = ','.join(bundle['features'])
            if workflow.family.estimation_attributes is not None:
                ds.setncatts(workflow.family.estimation_attributes(models))
            ds.lat, ds.lon = row['lat'], row['lon']
        temp.replace(output)
    finally:
        temp.unlink(missing_ok=True)
    return pixel, True


def estimate_pixels(workflow, area, pixels, workers):
    """Run ensemble inference for the rows in a pixel table."""
    workflow.estimation_directory(area).mkdir(parents=True, exist_ok=True)
    tasks = [(area, row) for row in pixels.to_dict('records')]
    with ProcessPoolExecutor(max_workers=min(workers, len(tasks)), mp_context=mp.get_context('spawn'),
                             initializer=partial(initialize_estimation, workflow)) as pool:
        return list(tqdm(pool.map(partial(estimate_pixel, workflow), tasks), total=len(tasks),
                         desc=f'{area} pixels', unit='pixel', leave=False))


def finish_estimations(workflow, area, results):
    saved = {pixel for pixel, success in results if success}
    folder = workflow.estimation_directory(area)
    for path in folder.glob('*_estimation.nc'):
        if path.name.removesuffix('_estimation.nc') not in saved:
            path.unlink()
    print(f'{area}: saved {len(saved)} of {len(results)} pixels '
          f'({len(results) - len(saved)} lack a complete event history)')


def build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--workers', type=int, default=40, help='Estimation processes (default: 40).')
    return parser


if __name__ == '__main__':
    args = build_argument_parser().parse_args()
    if args.workers < 1:
        raise ValueError('workers must be positive')
    workflow.load_ensemble()
    test_pixels = workflow.read_table('Test')
    results = estimate_pixels(workflow, 'Test', test_pixels, args.workers)
    finish_estimations(workflow, 'Test', results)
    print("05_LSTM_RZSM_estimation.py is finished.")
