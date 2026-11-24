#!/usr/bin/env python3
"""Step 2: EF pixel estimations from SMAP SSM with the SMAP T.

For every pixel in the Test pixel list, filter ``SMAP_SSM`` and write
``Exponential_Filter/RZSM_estimation/Test/<lat_idx>_<lon_idx>_estimation.nc``
with the in situ RZSM beside the estimation.
"""

import sys
from pathlib import Path
import argparse
from importlib import import_module
import multiprocessing as mp
import os

PUBLICATION_CODE = Path(__file__).resolve().parents[1]
if str(PUBLICATION_CODE) not in sys.path:
    sys.path.insert(0, str(PUBLICATION_CODE))

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

from core.project_paths import PATHS
from core.rzsm_pipeline_config import EF_POLICY, EF_SPIN_UP_DAYS, PIPELINE_VERSION, load_ef_t

exponential_filter = import_module("Exponential_Filter.01_EF_T_calibration").exponential_filter


AREA = 'Test'
DEFAULT_WORKERS = 60
SSM_VARIABLE = 'SMAP_SSM'
TARGET_VARIABLE = 'in-situ_RZSM'


def exponential_filter_from_ssm(ssm, raw_time, t_days):
    """Clean volumetric SSM, filter it, and mask the initial calendar spin-up."""
    values = np.asarray(ssm, dtype=float)
    dates = np.asarray(raw_time).astype('datetime64[ns]')
    if values.ndim != 1 or values.shape != dates.shape:
        raise ValueError('SSM and time must be aligned one-dimensional arrays.')
    if np.any(np.isnat(dates)):
        raise ValueError('Observation dates must not contain NaT.')
    values = values.copy()
    values[(values <= 0.0) | (values > 1.0)] = np.nan
    elapsed = ((dates - dates[0]) / np.timedelta64(1, 'D')).astype(float) if dates.size else np.array([])
    output = exponential_filter(values, elapsed, t_days)
    output[elapsed < EF_SPIN_UP_DAYS] = np.nan
    return output


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=DEFAULT_WORKERS, help='Parallel estimation processes (default: 60).')
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error('--workers must be positive')
    return args


def estimate_pixel(task):
    """Write one pixel's EF estimation file and return its path."""
    filename, input_dir, estimation_dir, t_days = task
    pixel_path = os.path.join(input_dir, filename)
    with xr.open_dataset(pixel_path) as ds:
        missing = [name for name in (SSM_VARIABLE, TARGET_VARIABLE) if name not in ds]
        if missing:
            raise KeyError(f"{pixel_path} is missing variables: {', '.join(missing)}")
        ssm = ds[SSM_VARIABLE].values.squeeze()
        rzsm = ds[TARGET_VARIABLE].values.squeeze()
        time = ds['time'].values

    estimation = exponential_filter_from_ssm(ssm, time, t_days)
    output = xr.Dataset(
        {
            'SSM': (['time'], ssm),
            'RZSM': (['time'], rzsm),
            'RZSM_estimation': (['time'], estimation),
        },
        coords={'time': time},
    )
    output.attrs.update({
        'source_estimation_variable': SSM_VARIABLE,
        'target_variable': TARGET_VARIABLE,
        'filter_T_days': t_days,
        'filter_policy': EF_POLICY,
        'spin_up_days': EF_SPIN_UP_DAYS,
        'pipeline_version': PIPELINE_VERSION,
    })

    output_path = os.path.join(estimation_dir, f"{filename.removesuffix('.nc')}_estimation.nc")
    temporary = f'{output_path}.tmp.{os.getpid()}'
    try:
        output.to_netcdf(temporary)
        os.replace(temporary, output_path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    return output_path


def listed_pixel_files(area, input_dir):
    """Return the pixel files of the area's pixel list, failing if any is missing."""
    pixel_list = (PATHS.results / 'International_Soil_Moisture_Network' / 'Station_TCA_Screening'
                  / f'Common_{area}_pixel_list.csv')
    if not pixel_list.is_file():
        raise FileNotFoundError(f'Pixel list was not found: {pixel_list}')
    pixels = pd.read_csv(pixel_list)
    if not {'lat_idx', 'lon_idx'}.issubset(pixels.columns):
        raise KeyError(f'{pixel_list} is missing columns lat_idx/lon_idx')
    if pixels.duplicated(['lat_idx', 'lon_idx']).any():
        raise RuntimeError(f'Duplicate pixel keys in {pixel_list}')
    filenames = sorted(f'{int(row.lat_idx)}_{int(row.lon_idx)}.nc' for row in pixels.itertuples())
    missing = [name for name in filenames if not os.path.exists(os.path.join(input_dir, name))]
    if missing:
        raise FileNotFoundError(f'{len(missing)} pixel files are missing from {input_dir}; '
                                f'first={missing[0]}')
    return filenames


def run(workers=DEFAULT_WORKERS):
    """Write the EF estimations of every listed Test pixel."""
    t_days = load_ef_t(PATHS.root, 'SMAP')
    area = AREA
    input_dir = PATHS.results / 'International_Soil_Moisture_Network' / f'ISMN_{area}'
    if not input_dir.is_dir():
        raise FileNotFoundError(f'ISMN input folder was not found: {input_dir}')
    filenames = listed_pixel_files(area, input_dir)

    estimation_dir = PATHS.results / 'Exponential_Filter' / 'RZSM_estimation' / area
    estimation_dir.mkdir(parents=True, exist_ok=True)
    for pattern in ('*_estimation.nc', '*_estimation.nc.tmp.*'):
        for stale in estimation_dir.glob(pattern):
            stale.unlink()

    tasks = [(name, os.fspath(input_dir), os.fspath(estimation_dir), t_days) for name in filenames]
    with mp.Pool(processes=workers) as pool:
        for _ in tqdm(pool.imap(estimate_pixel, tasks), total=len(tasks),
                      desc=f'EF estimation ({area})', leave=False):
            pass


def main(argv=None):
    args = parse_args(argv)
    run(workers=args.workers)


if __name__ == '__main__':
    main()
    print('02_EF_RZSM_estimation.py is finished.')
