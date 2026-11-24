'''Run selected SMAP/ASCAT-EF/land-model TCA triplets.

ASCAT EF fills slot 1 of every triplet and SMAP EF/FNO/LSTM slot 2; each EF
product must have been written with its satellite's T from
core.rzsm_pipeline_config.load_ef_t (02_EF_RZSM_estimation.py).
Omitting --rzsm runs all three SMAP models. Results are written to the canonical
Triple_Collocation_Analysis folders, replacing earlier ones. Each triplet's
stage stdout goes to Logs/ beside its results.
'''

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import multiprocessing as mp
import os
import subprocess
import sys
import tempfile
import warnings
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Iterable

PUBLICATION_ROOT = Path(__file__).resolve().parents[1]
if str(PUBLICATION_ROOT) not in sys.path:
    sys.path.insert(0, str(PUBLICATION_ROOT))

if __name__ in {'__main__', '__mp_main__'} and sys.argv[1:2] in (
    ['bootstrap'], ['mask'], ['average'], ['expand'],
):
    for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[variable] = '1'

import netCDF4
import numpy as np
from tqdm import tqdm

from core.project_paths import (
    DATA_ROOT, PATHS, estimation_output_root, existing_result_path,
    project_figures_root, project_results_root,
)
from core.rzsm_pipeline_config import (
    ANOMALY_WINDOW_DAYS,
    COMMON_START_DATE,
    NLDAS_TARGET_FILENAME,
    NLDAS_TARGET_VARIABLE,
    EF_SPIN_UP_DAYS,
    PIPELINE_VERSION,
    load_ef_t,
    TEST_PERIOD_POLICY,
    TEST_START_DATE,
    TEST_START_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
    test_period_mask,
)
from core.static_input_profiles import EVENT_FEATURES, EVENT_SCHEMA


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent

TCA_OPTIONS = ('default', 'amsr2', 'nldas')
EXPLICIT_MODELS = ('EF', 'FNO', 'LSTM')
EXPLICIT_LAND_PRODUCTS = ('ERA5-Land', 'NOAH', 'VIC', 'MOS')
FIXED_EF_MODEL_PAIRS = (
    ('EF', 'EF'),
    ('FNO', 'EF'),
    ('LSTM', 'EF'),
    ('EF', 'FNO'),
    ('EF', 'LSTM'),
)
TCA_ROOT_FOLDER = 'Triple_Collocation_Analysis'
TCA_SINGLE_RUN_FOLDER = 'Single_run'
TCA_BOOTSTRAP_FOLDER = 'Bootstrap'
TCA_SMAP_NN_FOLDER = 'SMAP-NN'
TCA_HDCA_FOLDER = 'HDCA'
TCA_BOOTSTRAP_COUNT = 5000
TCA_PERIOD_FOLDERS = {
    'full': 'Full_period',
    'test': 'Test_period',
}
ERA5_LAND_FILENAME = 'ERA5_Land_2015_2023_R_eqd_010.nc'
NLDAS_FILENAME = NLDAS_TARGET_FILENAME

NLDAS_MODEL_FILENAMES = {
    'NOAH': 'NLDAS_NOAH0125_H_daily_20150401_20231231_R_eqd_010_CONUS_target_7_28cm.nc',
    'VIC': 'NLDAS_VIC0125_H_daily_20150401_20231231_R_eqd_010_CONUS_target_7_28cm.nc',
    'MOS': 'NLDAS_MOS0125_H_daily_20150401_20231231_R_eqd_010_CONUS_target_7_28cm.nc',
}

NLDAS_MODEL_LABELS = {
    'NOAH': 'NLDAS Noah',
    'VIC': 'NLDAS VIC',
    'MOS': 'NLDAS Mosaic',
}

MODEL_DIRECTORIES = {
    'EF': 'Exponential_Filter',
    'FNO': 'Fourier_Neural_Operator',
    'LSTM': 'Long_Short-Term_Memory',
}


NEURAL_ESTIMATION_SCHEMA = "RZSM_static_ndvi_sand_precipitation_v1"
HISTORICAL_STATIC_FEATURES = ('NDVI_avg', 'Sand_RZSM', 'Precipitation_avg')
RECIPE_DIRECTORY = 'NDVI_Bulk_EventPrecipitation_N1'
NEURAL_STATIC_FEATURES = ",".join(HISTORICAL_STATIC_FEATURES)

_WORKER_D1 = None
_WORKER_D2 = None
_WORKER_D3 = None
_WORKER_BOOTSTRAP_INDICES = None
_WORKER_ETC = None
_WORKER_NOD_TH = None
_WORKER_CORR_TH = None

bounds = [-126, -66, 24, 51]
DEFAULT_TEST_START_RATIO = TEST_START_RATIO
SPINUP_DAYS = int(EF_SPIN_UP_DAYS)
BOOTSTRAP_COUNT = TCA_BOOTSTRAP_COUNT

DEFAULT_EF_ESTIMATION_FP = str(estimation_output_root(DATA_ROOT))
DEFAULT_NEURAL_ESTIMATION_FP = str(estimation_output_root())

nod_th = 25
corr_th = 0

SUPPORTED_MODELS = ['EF', 'FNO', 'LSTM']

SUPPORTED_DATA = ['ASCAT', 'SMAP']
SUPPORTED_ADD = ['SMAP_L4']

DEFAULT_WORKERS = 40
DEFAULT_RANDOM_SEED = 42
TCA_STAGES = ('calculation', 'masking', 'averaging', 'expansion')
PROGRESS_DESC_ENV = 'RZSM_TCA_PROGRESS_DESC'

LAND_PRODUCTS = ("ERA5-Land", "NOAH", "VIC", "MOS")
RZSM_MODELS = ("FNO", "LSTM", "EF")


def default_neural_estimation_root() -> Path:
    '''Match the shared estimation directory written by launcher 03.'''
    return estimation_output_root()


def require_fixed_ef_model_pair(
    ascat_model: str,
    smap_model: str,
) -> None:
    '''Reject a satellite-model pair that violates the fixed-opposite-EF policy.'''
    pair = (ascat_model, smap_model)
    if pair not in FIXED_EF_MODEL_PAIRS:
        raise ValueError(
            'The fixed-EF TCA structure requires SMAP-EF for an ASCAT '
            'target or ASCAT-EF for a SMAP target. '
            f'Allowed (ASCAT, SMAP) model pairs: {FIXED_EF_MODEL_PAIRS}'
        )


def fixed_ef_target_products(
    ascat_model: str,
    smap_model: str,
) -> tuple[str, ...]:
    '''
    Return the satellite/model products targeted by one fixed-EF triplet.

    ASCAT targets use SMAP-EF as the fixed opposite-sensor product, while
    SMAP targets use ASCAT-EF. The EF/EF pair therefore targets both products;
    every other allowed pair targets only its non-auxiliary satellite product.
    '''
    require_fixed_ef_model_pair(ascat_model, smap_model)

    targets = []
    if smap_model == 'EF':
        targets.append(f'ASCAT_{ascat_model}')
    if ascat_model == 'EF':
        targets.append(f'SMAP_{smap_model}')
    return tuple(targets)


def default_george_fp() -> Path:
    '''Identify the platform-specific root containing ``george_FP`` and ``Results``.'''
    return PATHS.root


def tca_root(george_fp: str | os.PathLike[str]) -> Path:
    return (
        project_results_root(Path(george_fp))
        / 'CONUS_RZSM_Estimation'
        / TCA_ROOT_FOLDER
    )


def single_run_analysis_root(
    george_fp: str | os.PathLike[str],
) -> Path:
    return tca_root(george_fp) / TCA_SINGLE_RUN_FOLDER


def single_run_figure_root(
    george_fp: str | os.PathLike[str],
) -> Path:
    '''Identify the canonical figure/CSV folder for deterministic TCA.'''
    return (
        project_figures_root(Path(george_fp))
        / 'CONUS_RZSM_Estimation'
        / TCA_ROOT_FOLDER
        / TCA_SINGLE_RUN_FOLDER
    )


def tca_figure_root(
    george_fp: str | os.PathLike[str],
) -> Path:
    '''Identify the figure/CSV folder holding one subfolder per TCA combination.'''
    return (
        project_figures_root(Path(george_fp))
        / 'CONUS_RZSM_Estimation'
        / TCA_ROOT_FOLDER
    )


def smap_nn_analysis_root(
    george_fp: str | os.PathLike[str],
) -> Path:
    return tca_root(george_fp) / TCA_SMAP_NN_FOLDER


def smap_nn_figure_root(
    george_fp: str | os.PathLike[str],
) -> Path:
    '''Identify the canonical publication-output folder for SMAP-NN TCA.'''
    return (
        project_figures_root(Path(george_fp))
        / 'CONUS_RZSM_Estimation'
        / TCA_ROOT_FOLDER
        / TCA_SMAP_NN_FOLDER
    )


def hdca_analysis_root(
    george_fp: str | os.PathLike[str],
) -> Path:
    '''Identify the analysis folder for the HDCA second-stage retrieval TCA.'''
    return tca_root(george_fp) / TCA_HDCA_FOLDER


def hdca_resampled_file(
    george_fp: str | os.PathLike[str],
) -> Path:
    '''Identify the 0.10-degree CONUS HDCA surface soil-moisture intermediate.'''
    return (
        hdca_analysis_root(george_fp)
        / 'Resampled'
        / 'HDCA_2nd_surface_eqd_010_CONUS.nc'
    )


def hdca_ef_file(
    george_fp: str | os.PathLike[str],
) -> Path:
    '''Identify the HDCA-EF root-zone product used as the TCA SMAP-family slot.'''
    return (
        hdca_analysis_root(george_fp)
        / 'Exponential_Filter'
        / 'EF_HDCA_2nd_estimation.nc'
    )


def ensure_tca_layout(
    george_fp: str | os.PathLike[str],
) -> dict[str, Path]:
    '''Create the canonical top-level TCA result folders.'''
    root = tca_root(george_fp)
    paths = {
        'single_run': root / TCA_SINGLE_RUN_FOLDER,
        'smap_nn': root / TCA_SMAP_NN_FOLDER,
        'hdca': root / TCA_HDCA_FOLDER,
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def analysis_folder(option: str, scheme: str) -> str:
    if option not in TCA_OPTIONS:
        raise ValueError(f'Unsupported TCA option: {option!r}')
    if scheme not in TCA_PERIOD_FOLDERS:
        raise ValueError(f'Unsupported TCA scheme: {scheme!r}')

    parts = [TCA_ROOT_FOLDER]
    if scheme != 'test':
        parts += [TCA_BOOTSTRAP_FOLDER, TCA_PERIOD_FOLDERS[scheme]]
    if option != 'default':
        parts.append(option.upper())
    return os.path.join(*parts)


def analysis_root(george_fp: str, option: str, scheme: str) -> str:
    return os.path.join(project_results_root(george_fp), 'CONUS_RZSM_Estimation', analysis_folder(option, scheme))


def tca_filename(model: str, data: str) -> str:
    return f'TCA_{model}_{data}.nc'


def explicit_analysis_root(george_fp: str, scheme: str) -> str:
    return analysis_root(george_fp, 'default', scheme)


def explicit_run_analysis_root(
    george_fp: str,
    scheme: str,
    run_name: str | None = None,
    period_folder: str | None = None,
) -> str:
    '''Identify the legacy bootstrap root or an isolated named-run root.'''
    anomaly_root = os.environ.get('RZSM_TCA_ANOMALY_ROOT')
    if anomaly_root:
        if scheme != 'test' or period_folder is None or not period_folder.startswith('Test_period_') or os.path.basename(period_folder) != period_folder:
            raise ValueError('RZSM_TCA_ANOMALY_ROOT requires a test-period sensitivity folder')
        root = os.path.join(anomaly_root, period_folder)
    elif period_folder is None or (
        scheme == 'test' and period_folder == TCA_PERIOD_FOLDERS['test']
    ):
        root = explicit_analysis_root(george_fp, scheme)
    else:
        if (
            scheme != 'test'
            or os.path.basename(period_folder) != period_folder
            or not (
                period_folder == TCA_PERIOD_FOLDERS['test']
                or period_folder.startswith('Test_period_')
            )
        ):
            raise ValueError(f'Invalid test-period folder: {period_folder!r}')
        root = os.path.join(project_results_root(george_fp), 'CONUS_RZSM_Estimation', TCA_ROOT_FOLDER, period_folder)
    if run_name is None:
        return root
    if not run_name or os.path.basename(run_name) != run_name or run_name in {'.', '..'}:
        raise ValueError(f'Invalid bootstrap run name: {run_name!r}')
    return os.path.join(root, 'Runs', run_name)


def available_ef_runs(
    george_fp: str | os.PathLike[str] | None = None,
) -> list[tuple[int, Path, Path | None]]:
    '''Locate the canonical TCA run for the calibrated SMAP timescale.'''

    root = Path(explicit_run_analysis_root(
        str(george_fp or default_george_fp()), 'test',
    ))
    merged = root / 'Merged'
    if not merged.is_dir():
        return []
    return [(int(load_ef_t(None, 'SMAP', required=True)), merged, None)]


def explicit_model3_token(model3: str) -> str:
    if model3 == 'ERA5-Land':
        return 'ERA5-Land'
    if model3 in NLDAS_MODEL_FILENAMES:
        return f'NLDAS_{model3}'
    raise ValueError(
        f'Unsupported land product: {model3!r}. '
        f'Available products: {EXPLICIT_LAND_PRODUCTS}'
    )


def explicit_tca_filename(model1: str, model2: str, model3: str) -> str:
    if model1 not in EXPLICIT_MODELS:
        raise ValueError(f'Unsupported ASCAT model: {model1!r}')
    if model2 not in EXPLICIT_MODELS:
        raise ValueError(f'Unsupported SMAP model: {model2!r}')
    model3_token = explicit_model3_token(model3)
    return f'TCA_{model1}_{model2}_{model3_token}.nc'


def canonical_product(path, folder):
    '''Identify canonical gridded outputs before enforcing their schema.'''
    canonical = project_results_root(Path(DATA_ROOT)) / 'CONUS_RZSM_Estimation' / folder
    return Path(path).resolve().is_relative_to(canonical.resolve())


def expected_schema(estimation_path):
    '''Select compatibility checks from the declared estimation recipe path.'''
    parts = Path(estimation_path).parts
    if RECIPE_DIRECTORY in parts:
        return EVENT_SCHEMA, ','.join(EVENT_FEATURES)
    if 'Estimations' in parts:
        relative_parts = parts[parts.index('Estimations') + 1:]
        if not relative_parts or relative_parts[0] in (
            'Fourier_Neural_Operator', 'Long_Short-Term_Memory',
        ):
            return EVENT_SCHEMA, ','.join(EVENT_FEATURES)
    return NEURAL_ESTIMATION_SCHEMA, NEURAL_STATIC_FEATURES


def require_estimation_dataset(dataset, path):
    '''Reject a canonical neural estimation made with another input recipe.'''
    if not canonical_product(path, "Estimations"):
        return
    schema, features = expected_schema(path)
    if getattr(dataset, "static_schema_version", None) != schema:
        raise ValueError(
            f"{path}: canonical TCA requires an FNO/LSTM estimation with the "
            f"schema {schema}; rerun the matching CONUS RZSM estimation script"
        )
    if getattr(dataset, "static_features", None) != features:
        raise ValueError(
            f"{path}: estimation static-feature order differs from the "
            f"recipe ({features})"
        )


def require_tca_dataset(dataset, path):
    '''Reject a canonical neural TCA output made from another estimation schema.'''
    if not canonical_product(path, "Triple_Collocation_Analysis/Merged"):
        return
    estimation_root = getattr(dataset, 'neural_estimation_root', '')
    schema, _ = expected_schema(estimation_root)
    if getattr(dataset, "model2", "") in ("FNO", "LSTM") and (
        getattr(dataset, "neural_static_schema_version", None)
        != schema
    ):
        raise ValueError(
            f"{path}: rerun canonical 05_TCA_calculation.py with the "
            f"matching neural estimations ({schema}) before plotting"
        )


def require_ready_product(path: str, label: str) -> None:
    '''Reject missing final products with a useful atomic-merge diagnostic.'''
    if os.path.isfile(path):
        if any(folder in Path(path).parts for folder in
               ('Fourier_Neural_Operator', 'Long_Short-Term_Memory')):
            with netCDF4.Dataset(path) as dataset:
                require_estimation_dataset(dataset, path)
        return

    directory, filename = os.path.split(path)
    temporary = os.path.join(directory, f'.{filename}.tmp')
    if os.path.isfile(temporary):
        raise FileNotFoundError(
            f'{label} final product is not ready yet: {path}. '
            f'The temporary merge file exists: {temporary}. Wait for its stack/merge '
            'process to finish; the temporary file will then be atomically renamed to '
            'the required final filename. Do not start TCA while that merge is running.'
        )
    raise FileNotFoundError(f'Missing {label} input product: {path}')


def estimation_product(
    estimation_root: str,
    model_directory: str,
    model: str,
    data: str,
) -> dict[str, object]:
    return {
        'path': os.fspath(existing_result_path(
            Path(estimation_root) / model_directory / f'{model}_{data}_estimation.nc'
        )),
        'variable': 'RZSM_avg' if model in {'FNO', 'LSTM'} else 'RZSM',
        'stem': f'RZSM_{model}_{data}',
        'label': f'{model}({data})',
        'allow_regrid': False,
    }


def explicit_estimation_product(
    estimation_root: str,
    model: str,
    data: str,
) -> dict[str, object]:
    if model not in MODEL_DIRECTORIES:
        raise ValueError(
            f'Unsupported {data} model: {model!r}. '
            f'Available models: {EXPLICIT_MODELS}'
        )
    return estimation_product(
        estimation_root,
        MODEL_DIRECTORIES[model],
        model,
        data,
    )


def nldas_model_product(
    george_fp: str,
    model: str,
) -> dict[str, object]:
    '''Identify one target-matched NLDAS land-surface-model product.'''
    if model not in NLDAS_MODEL_FILENAMES:
        raise ValueError(
            f'Unsupported NLDAS model: {model!r}. '
            f'Available models: {tuple(NLDAS_MODEL_FILENAMES)}'
        )

    label = NLDAS_MODEL_LABELS[model]
    return {
        'path': os.path.join(project_results_root(george_fp), 'North_American_Land_Data_Assimilation_System', NLDAS_MODEL_FILENAMES[model]),
        'variable': NLDAS_TARGET_VARIABLE,
        'stem': f'RZSM_NLDAS_{model}',
        'label': f'{label}(7-28 cm matched)',
        'allow_regrid': False,
    }


def nldas_product(george_fp: str) -> dict[str, object]:
    '''Legacy generic NLDAS product; preserve its historical Noah meaning.'''
    return nldas_model_product(george_fp, 'NOAH')


def auxiliary_product(
    george_fp: str,
    estimation_root: str,
    data: str,
    option: str,
) -> dict[str, object]:
    if option == 'default':
        other_data = 'SMAP' if data == 'ASCAT' else 'ASCAT'
        return {
            'path': os.path.join(
                estimation_root,
                'Exponential_Filter',
                f'EF_{other_data}_estimation.nc',
            ),
            'variable': 'RZSM',
            'stem': f'RZSM_EF_{other_data}',
            'label': f'EF({other_data})',
            'allow_regrid': False,
        }

    if option == 'amsr2':
        return {
            'path': os.path.join(
                estimation_root,
                'Exponential_Filter',
                'EF_AMSR2_estimation.nc',
            ),
            'variable': 'RZSM',
            'stem': 'RZSM_EF_AMSR2',
            'label': 'EF(AMSR2 nighttime)',
            'allow_regrid': False,
        }

    if option == 'nldas':
        return nldas_product(george_fp)

    raise ValueError(f'Unsupported TCA option: {option!r}')


def era5_product(george_fp: str) -> dict[str, object]:
    return {
        'path': os.path.join(project_results_root(george_fp), 'ECMWF_Reanalysis_v5_Land', ERA5_LAND_FILENAME),
        'variable': 'RZSM',
        'stem': 'RZSM_ERA5Land',
        'label': 'ERA5_Land',
        'allow_regrid': False,
    }


def land_product(george_fp: str, model3: str) -> dict[str, object]:
    if model3 == 'ERA5-Land':
        spec = era5_product(george_fp)
        spec['label'] = 'ERA5-Land'
        return spec
    if model3 in NLDAS_MODEL_FILENAMES:
        return nldas_model_product(george_fp, model3)
    raise ValueError(
        f'Unsupported land product: {model3!r}. '
        f'Available products: {EXPLICIT_LAND_PRODUCTS}'
    )


def explicit_triplet_label(model1: str, model2: str, model3: str) -> str:
    return f'{model1}(ASCAT) - {model2}(SMAP) - {model3}'


def triplet_label(model: str, data: str, option: str) -> str:
    if option == 'default':
        other_data = 'SMAP' if data == 'ASCAT' else 'ASCAT'
        auxiliary = f'EF({other_data})'
    elif option == 'amsr2':
        auxiliary = 'EF(AMSR2 nighttime)'
    elif option == 'nldas':
        auxiliary = 'NLDAS(7-28 cm matched)'
    else:
        raise ValueError(f'Unsupported TCA option: {option!r}')
    return f'{model}({data}) - {auxiliary} - ERA5_Land'


def cov_corr_three(X, Y, Z, calc_cov=True, calc_corr=True):
    results = {}
    X_2d = X.reshape(-1, X.shape[2])
    Y_2d = Y.reshape(-1, Y.shape[2])
    Z_2d = Z.reshape(-1, Z.shape[2])

    valid_mask = ~np.isnan(X_2d).all(axis=1) & ~np.isnan(Y_2d).all(axis=1) & ~np.isnan(Z_2d).all(axis=1)
    X_valid = X_2d[valid_mask]
    Y_valid = Y_2d[valid_mask]
    Z_valid = Z_2d[valid_mask]

    mean_X = np.nanmean(X_valid, axis=1, keepdims=True)
    mean_Y = np.nanmean(Y_valid, axis=1, keepdims=True)
    mean_Z = np.nanmean(Z_valid, axis=1, keepdims=True)

    X_centered = X_valid - mean_X
    Y_centered = Y_valid - mean_Y
    Z_centered = Z_valid - mean_Z

    n_valid_XY = np.sum(~np.isnan(X_valid) & ~np.isnan(Y_valid), axis=1)
    n_valid_YZ = np.sum(~np.isnan(Y_valid) & ~np.isnan(Z_valid), axis=1)
    n_valid_XZ = np.sum(~np.isnan(X_valid) & ~np.isnan(Z_valid), axis=1)

    original_shape = (X.shape[0], X.shape[1])

    if calc_cov:
        covXY = np.nansum(X_centered * Y_centered, axis=1) / (n_valid_XY - 1)
        covYZ = np.nansum(Y_centered * Z_centered, axis=1) / (n_valid_YZ - 1)
        covXZ = np.nansum(X_centered * Z_centered, axis=1) / (n_valid_XZ - 1)

        results['covXY'] = np.full(original_shape, np.nan)
        results['covYZ'] = np.full(original_shape, np.nan)
        results['covXZ'] = np.full(original_shape, np.nan)

        results['covXY'][valid_mask.reshape(original_shape)] = covXY
        results['covYZ'][valid_mask.reshape(original_shape)] = covYZ
        results['covXZ'][valid_mask.reshape(original_shape)] = covXZ

    if calc_corr:
        covXX = np.nansum(X_centered * X_centered, axis=1) / (n_valid_XY - 1)
        covYY = np.nansum(Y_centered * Y_centered, axis=1) / (n_valid_XY - 1)
        covZZ = np.nansum(Z_centered * Z_centered, axis=1) / (n_valid_YZ - 1)

        corrXY = covXY / np.sqrt(covXX * covYY)
        corrYZ = covYZ / np.sqrt(covYY * covZZ)
        corrXZ = covXZ / np.sqrt(covXX * covZZ)

        results['covXX'] = np.full(original_shape, np.nan)
        results['covYY'] = np.full(original_shape, np.nan)
        results['covZZ'] = np.full(original_shape, np.nan)
        results['corrXY'] = np.full(original_shape, np.nan)
        results['corrYZ'] = np.full(original_shape, np.nan)
        results['corrXZ'] = np.full(original_shape, np.nan)

        results['covXX'][valid_mask.reshape(original_shape)] = covXX
        results['covYY'][valid_mask.reshape(original_shape)] = covYY
        results['covZZ'][valid_mask.reshape(original_shape)] = covZZ
        results['corrXY'][valid_mask.reshape(original_shape)] = corrXY
        results['corrYZ'][valid_mask.reshape(original_shape)] = corrYZ
        results['corrXZ'][valid_mask.reshape(original_shape)] = corrXZ

    return results


def ETC_vec(X, Y, Z, nod_th=30, corr_th=0):
    '''Calculate vectorized extended triple collocation for bootstrap samples.'''

    combined_nan_mask = np.isnan(X) | np.isnan(Y) | np.isnan(Z)
    X[combined_nan_mask] = np.nan
    Y[combined_nan_mask] = np.nan
    Z[combined_nan_mask] = np.nan

    cov_corr_results = cov_corr_three(X, Y, Z)
    covXY = cov_corr_results.get('covXY')
    covXZ = cov_corr_results.get('covXZ')
    covYZ = cov_corr_results.get('covYZ')

    corrXY = cov_corr_results.get('corrXY')
    corrXZ = cov_corr_results.get('corrXZ')
    corrYZ = cov_corr_results.get('corrYZ')

    Xs = X
    Ys = Y
    Zs = Z
    X=[]; Y=[]; Z=[];

    cov_corr_results_s = cov_corr_three(Xs, Ys, Zs)

    covXXs = cov_corr_results_s.get('covXX')
    covYYs = cov_corr_results_s.get('covYY')
    covZZs = cov_corr_results_s.get('covZZ')
    covXYs = cov_corr_results_s.get('covXY')
    covXZs = cov_corr_results_s.get('covXZ')
    covYZs = cov_corr_results_s.get('covYZ')

    corrXYs = cov_corr_results_s.get('corrXYs')
    corrXZs = cov_corr_results_s.get('corrXZs')
    corrYZs = cov_corr_results_s.get('corrYZs')

    var_Xserr = covXXs - covXYs*covXZs/covYZs
    var_Yserr = covYYs - covXYs*covYZs/covXZs
    var_Zserr = covZZs - covXZs*covYZs/covXYs

    SNR_Xs =  (covXYs * covXZs / covYZs) / var_Xserr
    SNR_Ys =  (covXYs * covYZs / covXZs) / var_Yserr
    SNR_Zs =  (covXZs * covYZs / covXYs) / var_Zserr

    R_XYs = 1 / ((1 + 1/SNR_Xs) * (1 + 1/SNR_Ys)) ** 0.5
    R_XZs = 1 / ((1 + 1/SNR_Xs) * (1 + 1/SNR_Zs)) ** 0.5
    R_YZs = 1 / ((1 + 1/SNR_Ys) * (1 + 1/SNR_Zs)) ** 0.5

    fMSE_Xs = 1 / (1 + SNR_Xs)
    fMSE_Ys = 1 / (1 + SNR_Ys)
    fMSE_Zs = 1 / (1 + SNR_Zs)

    R_XXs = 1 / (1 + 1/SNR_Xs)
    R_YYs = 1 / (1 + 1/SNR_Ys)
    R_ZZs = 1 / (1 + 1/SNR_Zs)

    SNRdb_Xs = 10 * np.log10(SNR_Xs)
    SNRdb_Ys = 10 * np.log10(SNR_Ys)
    SNRdb_Zs = 10 * np.log10(SNR_Zs)

    VAR_err = {'x': var_Xserr, 'y': var_Yserr, 'z': var_Zserr}
    SNR = {'x': SNR_Xs, 'y': SNR_Ys, 'z': SNR_Zs}
    SNRdb = {'x': SNRdb_Xs, 'y': SNRdb_Ys, 'z': SNRdb_Zs}
    R = {'x': R_XXs, 'y': R_YYs, 'z': R_ZZs}
    fMSE = {'x': fMSE_Xs, 'y': fMSE_Ys, 'z': fMSE_Zs}

    condition_corr = (corrXY < corr_th) | (corrXZ < corr_th) | (corrYZ < corr_th)

    condition_n_valid = np.sum(~np.isnan(Xs) & ~np.isnan(Ys) & ~np.isnan(Zs), axis=2) < nod_th
    condition_fMSE = (fMSE_Xs < 0) | (fMSE_Ys < 0) | (fMSE_Zs < 0) | (fMSE_Xs > 1) | (fMSE_Ys > 1) | (fMSE_Zs > 1)
    condition_negative_vars_err  = (var_Xserr < 0) | (var_Yserr < 0) | (var_Zserr < 0)

    flags = {'condition_corr': condition_corr,
            'condition_n_valid': condition_n_valid,
            'condition_fMSE': condition_fMSE,
            'condition_negative_vars_err': condition_negative_vars_err}

    return VAR_err, SNR, SNRdb, R, fMSE, flags


def file_fingerprint(path: str | os.PathLike[str]) -> dict[str, object]:
    '''Identify inexpensive provenance that changes when an input is replaced.'''
    resolved = Path(path).resolve()
    stat = resolved.stat()
    return {
        'path': str(resolved),
        'size': int(stat.st_size),
        'mtime_ns': int(stat.st_mtime_ns),
    }


def fingerprints_json(paths: Iterable[str | os.PathLike[str]]) -> str:
    return json.dumps(
        [file_fingerprint(path) for path in paths],
        sort_keys=True,
        separators=(',', ':'),
    )


def build_bootstrap_indices(
    n_time: int,
    bootstrap_count: int,
    sample_size: int,
    random_seed: int,
    method: str = 'iid',
    block_length: int = 30,
) -> np.ndarray:
    if n_time <= 0:
        raise ValueError('n_time must be positive.')
    if bootstrap_count <= 0:
        raise ValueError('bootstrap_count must be positive.')
    if not 1 <= sample_size <= n_time:
        raise ValueError('sample_size must be between 1 and n_time.')
    if method not in {'moving_block', 'iid'}:
        raise ValueError(f'Unsupported bootstrap method: {method!r}')
    rng = np.random.default_rng(random_seed)
    if method == 'iid':
        return rng.integers(
            0,
            n_time,
            size=(bootstrap_count, sample_size),
            dtype=np.int32,
        )
    if not 1 <= int(block_length) <= n_time:
        raise ValueError('block_length must be between 1 and n_time.')
    block_length = int(block_length)
    blocks_per_sample = int(np.ceil(sample_size / block_length))
    maximum_start = n_time - block_length
    starts = rng.integers(
        0,
        maximum_start + 1,
        size=(bootstrap_count, blocks_per_sample),
        dtype=np.int32,
    )
    offsets = np.arange(block_length, dtype=np.int32)
    return (starts[:, :, None] + offsets[None, None, :]).reshape(
        bootstrap_count,
        -1,
    )[:, :sample_size]


def bootstrap_indices_sha256(indices: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(indices, dtype=np.int32)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def save_npy_atomic(path: str | os.PathLike[str], values: np.ndarray) -> None:
    path = Path(path)
    temporary = path.with_name(f'.{path.name}.tmp.npy')
    if temporary.exists():
        temporary.unlink()
    np.save(temporary, values)
    os.replace(temporary, path)


def write_rolling_anomaly(
    data: np.ndarray,
    path: str | os.PathLike[str],
    window_size: int,
    chunk_m: int,
    chunk_n: int,
) -> None:
    '''Write centered rolling anomalies without materializing another full cube.'''
    if window_size <= 1 or window_size % 2 == 0:
        raise ValueError('window_size must be an odd integer greater than 1.')
    source = np.asarray(data)
    if source.ndim != 3:
        raise ValueError('Anomaly input must have shape (lat, lon, time).')
    m, n, t = source.shape
    padding = window_size // 2
    starts = np.maximum(np.arange(t) - padding, 0)
    ends = np.minimum(np.arange(t) + padding + 1, t)

    path = Path(path)
    temporary = path.with_name(f'.{path.name}.tmp.npy')
    if temporary.exists():
        temporary.unlink()
    output = np.lib.format.open_memmap(
        temporary,
        mode='w+',
        dtype=np.float32,
        shape=source.shape,
    )
    try:
        for i0 in range(0, m, chunk_m):
            i1 = min(i0 + chunk_m, m)
            for j0 in range(0, n, chunk_n):
                j1 = min(j0 + chunk_n, n)
                chunk = np.asarray(source[i0:i1, j0:j1, :], dtype=np.float32)
                valid = np.isfinite(chunk)
                sums = np.cumsum(
                    np.where(valid, chunk, 0.0),
                    axis=-1,
                    dtype=np.float64,
                )
                counts = np.cumsum(valid, axis=-1, dtype=np.int32)
                sums = np.concatenate(
                    [np.zeros((*sums.shape[:2], 1)), sums],
                    axis=-1,
                )
                counts = np.concatenate(
                    [np.zeros((*counts.shape[:2], 1), dtype=np.int32), counts],
                    axis=-1,
                )
                window_sums = sums[:, :, ends] - sums[:, :, starts]
                window_counts = counts[:, :, ends] - counts[:, :, starts]
                rolling_mean = np.full(chunk.shape, np.nan, dtype=np.float32)
                np.divide(
                    window_sums,
                    window_counts,
                    out=rolling_mean,
                    where=window_counts > 0,
                )
                output[i0:i1, j0:j1, :] = chunk - rolling_mean
        output.flush()
    except BaseException:
        del output
        if temporary.exists():
            temporary.unlink()
        raise
    del output
    os.replace(temporary, path)


def write_triplet_support(
    path: str | os.PathLike[str],
    slot_arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
    trim: slice,
    time_axis: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    attrs: dict[str, object],
    nod_th: int,
) -> None:
    '''Write the evaluated-period common finite-date mask of one triplet.

    A centered anomaly is finite exactly where its source value is, so this is
    the date set every bootstrap replicate resamples from. Raw/ stores only
    whether a replicate fell below nod_th, not the count itself.
    '''
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    m, n = slot_arrays[0].shape[:2]
    t = len(time_axis)
    common = np.ones((m, n, t), dtype=bool)
    slot_counts = []
    for values in slot_arrays:
        finite = np.isfinite(values[:, :, trim])
        if finite.shape != common.shape:
            raise ValueError(f'Support slice {finite.shape} differs from {common.shape}.')
        slot_counts.append(finite.sum(axis=2, dtype=np.int32))
        common &= finite

    time_values = np.asarray(time_axis)
    if np.issubdtype(time_values.dtype, np.datetime64):
        days = time_values.astype('datetime64[D]').astype(np.int64)
        time_units = 'days since 1970-01-01'
    else:
        days = np.arange(t, dtype=np.int64)
        time_units = 'index within the evaluated period'

    temporary = path.with_name(f'.{path.name}.tmp.{os.getpid()}')
    if temporary.exists():
        temporary.unlink()
    try:
        with netCDF4.Dataset(temporary, 'w', format='NETCDF4') as output:
            output.createDimension('lat', m)
            output.createDimension('lon', n)
            output.createDimension('time', t)
            for name, values, units in (
                ('lat', lat, 'degrees_north'), ('lon', lon, 'degrees_east'),
            ):
                variable = output.createVariable(name, 'f4', (name,))
                variable[:] = np.asarray(values, dtype=np.float32)
                variable.units = units
            time_variable = output.createVariable('time', 'i4', ('time',))
            time_variable[:] = days.astype(np.int32)
            time_variable.units = time_units
            mask = output.createVariable(
                'common_valid', 'u1', ('lat', 'lon', 'time'),
                zlib=True, complevel=4, shuffle=True,
                chunksizes=(min(20, m), min(20, n), t),
            )
            mask[:] = common.astype(np.uint8)
            mask.long_name = 'all three triplet members finite on this evaluated date'
            count = output.createVariable('n_common_dates', 'i2', ('lat', 'lon'), zlib=True)
            count[:] = common.sum(axis=2).astype(np.int16)
            for index, values in enumerate(slot_counts, start=1):
                variable = output.createVariable(
                    f'n_valid_slot{index}', 'i2', ('lat', 'lon'), zlib=True,
                )
                variable[:] = values.astype(np.int16)
            for name, value in attrs.items():
                if value is not None:
                    output.setncattr(name, value)
            output.nod_th = int(nod_th)
            output.processing_stage = 'Support'
            output.run_complete = 1
        os.replace(temporary, path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def initialize_worker(
    anomaly_paths: tuple[str, str, str],
    bootstrap_indices_path: str,
    nod_th: int,
    corr_th: int,
) -> None:
    '''Open read-only memmaps in each worker instead of copying multi-GB arrays.'''
    global _WORKER_D1, _WORKER_D2, _WORKER_D3
    global _WORKER_BOOTSTRAP_INDICES, _WORKER_ETC
    global _WORKER_NOD_TH, _WORKER_CORR_TH


    _WORKER_D1 = np.load(anomaly_paths[0], mmap_mode='r')
    _WORKER_D2 = np.load(anomaly_paths[1], mmap_mode='r')
    _WORKER_D3 = np.load(anomaly_paths[2], mmap_mode='r')
    _WORKER_BOOTSTRAP_INDICES = np.load(
        bootstrap_indices_path,
        mmap_mode='r',
    )
    _WORKER_ETC = ETC_vec
    _WORKER_NOD_TH = int(nod_th)
    _WORKER_CORR_TH = int(corr_th)


def process_bootstrap_batch(task: tuple[int, int, int, int, int, int]):
    '''Calculate one spatial chunk for a bounded range of bootstraps.'''
    boot_start, boot_end, i0, i1, j0, j1 = task
    batch_count = boot_end - boot_start
    spatial_shape = (i1 - i0, j1 - j0, batch_count)
    outputs = [
        np.full(spatial_shape, np.nan, dtype=np.float32)
        for _ in range(10)
    ]

    d1_chunk = _WORKER_D1[i0:i1, j0:j1, :]
    d2_chunk = _WORKER_D2[i0:i1, j0:j1, :]
    d3_chunk = _WORKER_D3[i0:i1, j0:j1, :]

    for local_index, boot_index in enumerate(range(boot_start, boot_end)):
        time_indices = _WORKER_BOOTSTRAP_INDICES[boot_index]
        d1 = np.take(d1_chunk, time_indices, axis=-1)
        d2 = np.take(d2_chunk, time_indices, axis=-1)
        d3 = np.take(d3_chunk, time_indices, axis=-1)

        with np.errstate(divide='ignore', invalid='ignore'):
            var_err, snr, _, _, _, flags = _WORKER_ETC(
                d1,
                d2,
                d3,
                nod_th=_WORKER_NOD_TH,
                corr_th=_WORKER_CORR_TH,
            )
        batch_values = (
            var_err['x'],
            var_err['y'],
            var_err['z'],
            snr['x'],
            snr['y'],
            snr['z'],
            flags['condition_corr'],
            flags['condition_n_valid'],
            flags['condition_fMSE'],
            flags['condition_negative_vars_err'],
        )
        for output, values in zip(outputs, batch_values):
            output[:, :, local_index] = np.asarray(values, dtype=np.float32)

    return (boot_start, boot_end, i0, i1, j0, j1, *outputs)


def iter_bootstrap_tasks(
    spatial_chunks: Iterable[tuple[int, int, int, int]],
    bootstrap_count: int,
    bootstrap_batch_size: int,
):
    if bootstrap_batch_size <= 0:
        raise ValueError('bootstrap_batch_size must be positive.')
    for boot_start in range(0, bootstrap_count, bootstrap_batch_size):
        boot_end = min(boot_start + bootstrap_batch_size, bootstrap_count)
        for i0, i1, j0, j1 in spatial_chunks:
            yield boot_start, boot_end, i0, i1, j0, j1


def task_count(
    spatial_chunk_count: int,
    bootstrap_count: int,
    bootstrap_batch_size: int,
) -> int:
    batches = (bootstrap_count + bootstrap_batch_size - 1) // bootstrap_batch_size
    return int(spatial_chunk_count * batches)


def create_raw_dataset(
    path: str | os.PathLike[str],
    lat: np.ndarray,
    lon: np.ndarray,
    bootstrap_indices: np.ndarray,
    variable_names: tuple[str, ...],
    attrs: dict[str, object],
    chunk_m: int,
    chunk_n: int,
    bootstrap_batch_size: int,
) -> netCDF4.Dataset:
    if len(variable_names) != 10:
        raise ValueError('Exactly ten TCA result variables are required.')

    dataset = netCDF4.Dataset(path, 'w', format='NETCDF4')
    dataset.createDimension('lat', len(lat))
    dataset.createDimension('lon', len(lon))
    dataset.createDimension('boot', bootstrap_indices.shape[0])
    dataset.createDimension('sample', bootstrap_indices.shape[1])

    lat_var = dataset.createVariable('lat', 'f4', ('lat',))
    lon_var = dataset.createVariable('lon', 'f4', ('lon',))
    lat_var[:] = np.asarray(lat, dtype=np.float32)
    lon_var[:] = np.asarray(lon, dtype=np.float32)
    lat_var.units = 'degrees_north'
    lon_var.units = 'degrees_east'

    index_var = dataset.createVariable(
        'bootstrap_time_index',
        'i4',
        ('boot', 'sample'),
        zlib=True,
        complevel=1,
        chunksizes=(
            min(bootstrap_batch_size, bootstrap_indices.shape[0]),
            bootstrap_indices.shape[1],
        ),
    )
    index_var[:] = np.asarray(bootstrap_indices, dtype=np.int32)
    index_var.long_name = 'source time indices sampled with replacement'

    chunksizes = (
        min(chunk_m, len(lat)),
        min(chunk_n, len(lon)),
        min(bootstrap_batch_size, bootstrap_indices.shape[0]),
    )
    for name in variable_names:
        dataset.createVariable(
            name,
            'f4',
            ('lat', 'lon', 'boot'),
            zlib=True,
            complevel=4,
            fill_value=np.float32(np.nan),
            chunksizes=chunksizes,
        )

    for name, value in attrs.items():
        if value is None:
            continue
        dataset.setncattr(name, value)
    dataset.run_complete = 0
    return dataset


def write_bootstrap_batch(
    dataset: netCDF4.Dataset,
    variable_names: tuple[str, ...],
    result,
) -> None:
    boot_start, boot_end, i0, i1, j0, j1, *values = result
    destination = np.s_[i0:i1, j0:j1, boot_start:boot_end]
    for name, array in zip(variable_names, values):
        dataset.variables[name][destination] = array


def parse_bootstrap_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run one explicit test-period temporal stability-bootstrap TCA."
    )
    parser.add_argument('--model1', required=True, choices=EXPLICIT_MODELS)
    parser.add_argument('--model2', required=True, choices=EXPLICIT_MODELS)
    parser.add_argument('--model3', required=True, choices=EXPLICIT_LAND_PRODUCTS)
    parser.add_argument('--run_name', default=None)
    parser.add_argument('--period_folder', default=None)
    parser.add_argument('--ef_estimation_root', default=DEFAULT_EF_ESTIMATION_FP)
    parser.add_argument(
        '--neural_estimation_root', default=DEFAULT_NEURAL_ESTIMATION_FP
    )
    parser.add_argument('--test_start_ratio', type=float, default=DEFAULT_TEST_START_RATIO, help='Start ratio within the post-spinup period; default keeps the final 20 percent')
    parser.add_argument('--anomaly_window_days', type=int, default=ANOMALY_WINDOW_DAYS)
    parser.add_argument('--test_end_date', default=None, help='Inclusive YYYY-MM-DD evaluation end; later dates remain anomaly context')
    parser.add_argument('--n_cores', type=int, default=40, help='Number of multiprocessing workers for TCA')
    parser.add_argument('--chunk_m', type=int, default=20, help='Latitude chunk size for TCA')
    parser.add_argument('--chunk_n', type=int, default=20, help='Longitude chunk size for TCA')
    parser.add_argument('--random_seed', type=int, default=42, help='Seed for reproducible temporal bootstrap sampling')
    parser.add_argument('--bootstrap_method', choices=['moving_block', 'iid'], default='moving_block')
    parser.add_argument('--block_length', type=int, default=30)
    parser.add_argument('--bootstrap_fraction', type=float, default=1.0)
    parser.add_argument('--bootstrap_count', type=int, default=TCA_BOOTSTRAP_COUNT)
    parser.add_argument('--bootstrap_batch_size', type=int, default=25, help='Bootstraps calculated per spatial worker task')
    parser.add_argument(
        '--slot2_file',
        default=None,
        help='Replace the SMAP-slot product with this NetCDF file.',
    )
    parser.add_argument(
        '--slot2_variable',
        default='RZSM',
        help='Variable name inside --slot2_file.',
    )
    parser.add_argument(
        '--slot2_stem',
        default=None,
        help='Output variable-name stem for the substituted SMAP slot.',
    )
    parser.add_argument(
        '--slot2_label',
        default=None,
        help='Human-readable label for the substituted SMAP slot.',
    )
    parser.add_argument(
        '--support_output',
        default=None,
        help=(
            'Write the triplet common finite-date mask for the evaluated period '
            'to this NetCDF file and exit without bootstrapping.'
        ),
    )
    args = parser.parse_args(argv)

    require_fixed_ef_model_pair(args.model1, args.model2)
    if not 0.0 <= args.test_start_ratio <= 1.0:
        raise ValueError("--test_start_ratio must be between 0 and 1.")
    if args.n_cores <= 0:
        raise ValueError("--n_cores must be positive.")
    if args.chunk_m <= 0 or args.chunk_n <= 0:
        raise ValueError("--chunk_m and --chunk_n must be positive.")
    if args.bootstrap_batch_size <= 0:
        raise ValueError("--bootstrap_batch_size must be positive.")
    if args.random_seed < 0:
        raise ValueError("--random_seed must be non-negative.")
    if args.block_length <= 0:
        raise ValueError("--block_length must be positive.")
    if not 0.0 < args.bootstrap_fraction <= 1.0:
        raise ValueError("--bootstrap_fraction must be in (0, 1].")
    if args.bootstrap_count <= 0:
        raise ValueError("--bootstrap_count must be positive.")
    if args.anomaly_window_days <= 0 or args.anomaly_window_days % 2 == 0:
        raise ValueError("--anomaly_window_days must be a positive odd number.")
    if args.test_end_date is not None:
        try:
            np.datetime64(args.test_end_date, 'D')
        except ValueError as exc:
            raise ValueError("--test_end_date must use YYYY-MM-DD format.") from exc
    if args.slot2_file is not None:
        if not args.slot2_stem or not args.slot2_label:
            raise ValueError(
                "--slot2_file requires both --slot2_stem and --slot2_label."
            )
        if args.model2 != 'EF':
            raise ValueError(
                "--slot2_file substitutes the SMAP slot itself, so --model2 must "
                "be EF; the substituted product carries its own label."
            )
    return args


def fill_masked_nan(values):
    return np.ma.filled(values, np.nan).astype(np.float32)


def estimation_metadata(path, family):
    with netCDF4.Dataset(path, 'r') as source:
        if family in ('FNO', 'LSTM'):
            require_estimation_dataset(source, path)
        if int(getattr(source, 'run_complete', 0)) != 1:
            raise RuntimeError(f'Estimation input is incomplete: {path}')
        attr_name = f'{family}_model'
        if attr_name in source.ncattrs():
            return source.getncattr(attr_name)
        if getattr(source, 'estimation_model', None) == family:
            return family
        raise KeyError(f'{path} has neither {attr_name} nor estimation_model = {family!r}')


def estimation_root(model, args):
    return (
        args.ef_estimation_root
        if model == 'EF'
        else args.neural_estimation_root
    )


def region_slices(lat_1d, lon_1d, bounds):
    lon_min, lon_max, lat_min, lat_max = bounds
    lat_idx = np.where((lat_1d >= lat_min) & (lat_1d <= lat_max))[0]
    lon_idx = np.where((lon_1d >= lon_min) & (lon_1d <= lon_max))[0]

    if len(lat_idx) == 0 or len(lon_idx) == 0:
        raise ValueError(f"No grid cells found inside bounds {bounds}.")

    return (
        slice(int(lat_idx.min()), int(lat_idx.max()) + 1),
        slice(int(lon_idx.min()), int(lon_idx.max()) + 1)
    )


def read_time_axis(ds):
    if 'time' not in ds.variables:
        return None

    time_var = ds.variables['time']
    values = time_var[:]
    units = getattr(time_var, 'units', None)
    calendar = getattr(time_var, 'calendar', 'standard')

    if units is None:
        return np.asarray(values)

    unit_name = units.split('since', 1)[0].strip().lower() if 'since' in units else ''
    if unit_name in {'nanosecond', 'nanoseconds', 'ns'}:
        origin_text = units.split('since', 1)[1].strip().replace(' ', 'T')
        origin = np.datetime64(origin_text, 'ns')
        elapsed = np.asarray(values).astype(np.int64).astype('timedelta64[ns]')
        return (origin + elapsed).astype('datetime64[D]')

    try:
        dates = netCDF4.num2date(
            values,
            units=units,
            calendar=calendar,
            only_use_cftime_datetimes=False,
            only_use_python_datetimes=True
        )
    except TypeError:
        dates = netCDF4.num2date(values, units=units, calendar=calendar)

    return np.asarray(
        [np.datetime64(f"{d.year:04d}-{d.month:02d}-{d.day:02d}", 'D') for d in dates],
        dtype='datetime64[D]'
    )


def find_variable_name(nc, candidates, label, file_path):
    for name in candidates:
        if name in nc.variables:
            return name
    raise KeyError(
        f"{label} coordinate not found in {file_path}. "
        f"Tried {', '.join(candidates)}."
    )


def load_rzsm_region(file_path, var_name, lat_names, lon_names, label):
    require_ready_product(file_path, label)
    with netCDF4.Dataset(file_path) as nc:
        lat_name = find_variable_name(nc, lat_names, 'Latitude', file_path)
        lon_name = find_variable_name(nc, lon_names, 'Longitude', file_path)
        lat_1d = np.asarray(nc[lat_name][:], dtype=np.float32)
        lon_1d = np.asarray(nc[lon_name][:], dtype=np.float32)
        lat_slice, lon_slice = region_slices(lat_1d, lon_1d, bounds)

        var = nc[var_name]
        lat_dim = nc[lat_name].dimensions[0]
        lon_dim = nc[lon_name].dimensions[0]
        index = []
        retained_axes = []
        for dim in var.dimensions:
            if dim == lat_dim:
                index.append(lat_slice)
                retained_axes.append('lat')
            elif dim == lon_dim:
                index.append(lon_slice)
                retained_axes.append('lon')
            elif dim == 'time':
                index.append(slice(None))
                retained_axes.append('time')
            elif len(nc.dimensions[dim]) == 1:
                index.append(0)
            else:
                raise ValueError(
                    f'{label} variable {var_name!r} has unsupported dimension {dim!r} '
                    f'in {file_path}.'
                )
        if set(retained_axes) != {'lat', 'lon', 'time'}:
            raise ValueError(
                f'{label} variable {var_name!r} must contain latitude, longitude, and time; '
                f'found dimensions {var.dimensions}.'
            )
        raw = fill_masked_nan(var[tuple(index)])
        data = np.transpose(
            raw,
            tuple(retained_axes.index(name) for name in ('lat', 'lon', 'time')),
        )
        time_axis = read_time_axis(nc)

    lat_region = lat_1d[lat_slice]
    lon_region = lon_1d[lon_slice]

    return data, lat_region, lon_region, time_axis


def align_axis_to_reference(data, reference_axis, axis_values, label, data_axis, atol=1e-4):
    reference_axis = np.asarray(reference_axis)
    axis_values = np.asarray(axis_values)

    if reference_axis.shape != axis_values.shape:
        raise ValueError(
            f"{label} axis shape {axis_values.shape} does not match reference axis {reference_axis.shape}."
        )

    direct_diff = np.nanmax(np.abs(reference_axis - axis_values))
    if direct_diff <= atol:
        return data, axis_values

    reversed_values = axis_values[::-1]
    reversed_diff = np.nanmax(np.abs(reference_axis - reversed_values))
    if reversed_diff <= atol:
        return np.flip(data, axis=data_axis), reversed_values

    raise ValueError(
        f"{label} axis is not aligned to reference axis: "
        f"max direct diff={direct_diff}, max reversed diff={reversed_diff}."
    )


def align_grid_to_reference(data, reference_lat, reference_lon, lat_grid, lon_grid, label):
    data, lat_grid = align_axis_to_reference(data, reference_lat, lat_grid, label, data_axis=0)
    data, lon_grid = align_axis_to_reference(data, reference_lon, lon_grid, label, data_axis=1)
    return data, lat_grid, lon_grid


def regrid_bilinear_valid(data, source_lat, source_lon, target_lat, target_lon, time_chunk=30):
    '''Bilinearly map a regular grid while renormalizing around missing neighbors.'''
    source_lat = np.asarray(source_lat, dtype=np.float64)
    source_lon = np.asarray(source_lon, dtype=np.float64)
    target_lat = np.asarray(target_lat, dtype=np.float64)
    target_lon = np.asarray(target_lon, dtype=np.float64)
    source = np.asarray(data, dtype=np.float32)

    if source_lat[0] > source_lat[-1]:
        source_lat = source_lat[::-1]
        source = source[::-1, :, :]
    if source_lon[0] > source_lon[-1]:
        source_lon = source_lon[::-1]
        source = source[:, ::-1, :]
    if np.any(np.diff(source_lat) <= 0) or np.any(np.diff(source_lon) <= 0):
        raise ValueError('Source grid must have unique monotonic latitude/longitude axes.')

    def brackets(source_axis, target_axis):
        upper = np.searchsorted(source_axis, target_axis, side='right')
        upper = np.clip(upper, 1, len(source_axis) - 1)
        lower = upper - 1
        fraction = (
            (target_axis - source_axis[lower])
            / (source_axis[upper] - source_axis[lower])
        )
        inside = (target_axis >= source_axis[0]) & (target_axis <= source_axis[-1])
        return lower, upper, np.clip(fraction, 0.0, 1.0), inside

    lat0, lat1, lat_fraction, lat_inside = brackets(source_lat, target_lat)
    lon0, lon1, lon_fraction, lon_inside = brackets(source_lon, target_lon)
    inside = lat_inside[:, None] & lon_inside[None, :]
    weights = (
        (1.0 - lat_fraction)[:, None] * (1.0 - lon_fraction)[None, :],
        (1.0 - lat_fraction)[:, None] * lon_fraction[None, :],
        lat_fraction[:, None] * (1.0 - lon_fraction)[None, :],
        lat_fraction[:, None] * lon_fraction[None, :],
    )

    output = np.full(
        (len(target_lat), len(target_lon), source.shape[2]),
        np.nan,
        dtype=np.float32,
    )
    for start in range(0, source.shape[2], time_chunk):
        end = min(start + time_chunk, source.shape[2])
        neighbors = (
            source[lat0[:, None], lon0[None, :], start:end],
            source[lat0[:, None], lon1[None, :], start:end],
            source[lat1[:, None], lon0[None, :], start:end],
            source[lat1[:, None], lon1[None, :], start:end],
        )
        numerator = np.zeros(neighbors[0].shape, dtype=np.float32)
        denominator = np.zeros(neighbors[0].shape, dtype=np.float32)
        for values, spatial_weight in zip(neighbors, weights):
            valid = np.isfinite(values)
            expanded_weight = spatial_weight[:, :, None].astype(np.float32)
            numerator += np.where(valid, values * expanded_weight, 0.0)
            denominator += np.where(valid, expanded_weight, 0.0)
        mapped = np.full(numerator.shape, np.nan, dtype=np.float32)
        np.divide(numerator, denominator, out=mapped, where=denominator > 0.0)
        mapped[~inside, :] = np.nan
        output[:, :, start:end] = mapped
    return output


def align_or_regrid_to_reference(
    data,
    reference_lat,
    reference_lon,
    lat_grid,
    lon_grid,
    label,
    allow_regrid=False,
):
    try:
        return align_grid_to_reference(
            data, reference_lat, reference_lon, lat_grid, lon_grid, label
        )
    except ValueError:
        if not allow_regrid:
            raise
    mapped = regrid_bilinear_valid(
        data, lat_grid, lon_grid, reference_lat, reference_lon
    )
    return mapped, np.asarray(reference_lat), np.asarray(reference_lon)


def grid_orientation(lat, lon):
    lat_order = 'ascending' if lat[0] <= lat[-1] else 'descending'
    lon_order = 'ascending' if lon[0] <= lon[-1] else 'descending'
    return f'latitude {lat_order}, longitude {lon_order}'


def time_keys(time_axis):
    if time_axis is None:
        return None

    values = np.asarray(time_axis)
    if np.issubdtype(values.dtype, np.datetime64):
        return values.astype('datetime64[D]').astype(np.int64)
    if np.issubdtype(values.dtype, np.number):
        return np.rint(values.astype(np.float64)).astype(np.int64)
    return values.astype(str)


def common_time_index(T1, T2, T3, D1, D2, D3):
    K1 = time_keys(T1)
    K2 = time_keys(T2)
    K3 = time_keys(T3)

    if K1 is None or K2 is None or K3 is None:
        min_len = min(D1.shape[2], D2.shape[2], D3.shape[2])
        common_time = np.arange(min_len, dtype=np.int64) if T1 is None else np.asarray(T1)[:min_len]
        return (
            D1[:, :, :min_len],
            D2[:, :, :min_len],
            D3[:, :, :min_len],
            common_time
        )

    common_keys = np.intersect1d(np.intersect1d(K1, K2), K3)
    if len(common_keys) == 0:
        raise ValueError("No common time steps were found across the three TCA inputs.")

    index1 = {key: idx for idx, key in enumerate(K1)}
    index2 = {key: idx for idx, key in enumerate(K2)}
    index3 = {key: idx for idx, key in enumerate(K3)}

    idx1 = np.asarray([index1[key] for key in common_keys], dtype=np.int64)
    idx2 = np.asarray([index2[key] for key in common_keys], dtype=np.int64)
    idx3 = np.asarray([index3[key] for key in common_keys], dtype=np.int64)
    common_time = np.asarray(T1)[idx1]

    return D1[:, :, idx1], D2[:, :, idx2], D3[:, :, idx3], common_time


def select_test_period(
    D1, D2, D3, common_time, spinup_days, test_start_ratio,
    anomaly_window_days, test_end_date=None,
):
    n_time = len(common_time)
    common_values = np.asarray(common_time)
    if np.issubdtype(common_values.dtype, np.datetime64):
        period_mask = test_period_mask(common_values)
        if test_end_date is not None:
            period_mask &= (
                common_values.astype('datetime64[D]')
                <= np.datetime64(test_end_date, 'D')
            )
        keep = np.flatnonzero(period_mask)
        test_start = int(keep[0]) if len(keep) else n_time
        spinup_start = int(np.searchsorted(
            common_values.astype('datetime64[D]'),
            COMMON_START_DATE,
            side='left',
        ))
        post_spinup_steps = n_time - spinup_start
        selection_method = TEST_PERIOD_POLICY
    else:
        spinup_start = min(max(int(spinup_days), 0), n_time)
        post_spinup_steps = n_time - spinup_start
        test_start = spinup_start + int(post_spinup_steps * float(test_start_ratio))
        test_start = min(max(test_start, spinup_start), n_time)
        keep = np.arange(test_start, n_time)
        selection_method = 'ratio_fallback_for_undecoded_time'
    if len(keep) <= nod_th:
        raise ValueError(
            f"Only {len(keep)} test-period time steps remain after spin-up and split; "
            f"need more than nod_th={nod_th}."
        )

    metadata = {
        'aligned_full_time_start': str(common_time[0]),
        'aligned_full_time_end': str(common_time[-1]),
        'aligned_full_time_steps': int(n_time),
        'post_spinup_time_steps': int(post_spinup_steps),
        'test_start_index_after_alignment': int(test_start),
        'test_period_selection': selection_method,
    }

    pad = int(anomaly_window_days) // 2
    pad_start = max(0, int(keep[0]) - pad)
    pad_end = min(n_time, int(keep[-1]) + 1 + pad)
    expected_start = int(keep[0]) - pad
    expected_end = int(keep[-1]) + 1 + pad
    natural_record_end = int(keep[-1]) == n_time - 1
    context_complete = expected_start >= 0 and (
        expected_end <= n_time or natural_record_end
    )
    if not context_complete:
        raise ValueError(
            "The evaluation interval lacks centered-anomaly context for "
            f"a {anomaly_window_days}-day window. Supply an earlier "
            "--test_end_date (and, if necessary, a later test start). "
            "Truncated asymmetric anomaly windows are not permitted."
        )
    context = np.arange(pad_start, pad_end)
    trim = slice(int(keep[0]) - pad_start, int(keep[0]) - pad_start + len(keep))

    metadata.update(
        {
            'anomaly_context_pad_days': int(pad),
            'anomaly_context_time_steps': int(len(context)),
            'anomaly_context_complete_both_sides': int(
                expected_start >= 0 and expected_end <= n_time
            ),
            'anomaly_right_edge_uses_available_record': int(
                natural_record_end and expected_end > n_time
            ),
            'anomaly_context_policy': (
                'anomaly removed on the test period plus available window//2 '
                'days of context, then trimmed; equals a full-record anomaly '
                'restricted to the test period, including its natural edge'
            ),
        }
    )

    return (
        D1[:, :, context],
        D2[:, :, context],
        D3[:, :, context],
        common_time[keep],
        trim,
        metadata,
    )


def align_triplet_test_period(
    D1, T1, D2, T2, D3, T3, spinup_days, test_start_ratio,
    anomaly_window_days, test_end_date=None,
):
    D1, D2, D3, common_time = common_time_index(T1, T2, T3, D1, D2, D3)
    return select_test_period(
        D1, D2, D3, common_time, spinup_days, test_start_ratio,
        anomaly_window_days, test_end_date,
    )


def make_spatial_chunks(m, n, chunk_m=20, chunk_n=20):
    chunks = []
    for i0 in range(0, m, chunk_m):
        i1 = min(i0 + chunk_m, m)
        for j0 in range(0, n, chunk_n):
            j1 = min(j0 + chunk_n, n)
            chunks.append((i0, i1, j0, j1))
    return chunks


def bootstrap_main(argv=None):
    args = parse_bootstrap_args(argv)
    model1, model2, model3 = args.model1, args.model2, args.model3
    slot1_spec = explicit_estimation_product(
        estimation_root(model1, args), model1, 'ASCAT'
    )
    slot2_spec = (
        {
            'path': os.path.abspath(args.slot2_file),
            'variable': args.slot2_variable,
            'stem': args.slot2_stem,
            'label': args.slot2_label,
            'allow_regrid': False,
        }
        if args.slot2_file is not None
        else explicit_estimation_product(
            estimation_root(model2, args), model2, 'SMAP'
        )
    )
    ef_t_days = {}
    for slot_model, slot_data, slot_spec in (
        (model1, 'ASCAT', slot1_spec),
        (model2, 'SMAP', slot2_spec if args.slot2_file is None else None),
    ):
        if slot_model != 'EF' or slot_spec is None:
            continue
        expected_t = load_ef_t(None, slot_data, required=True)
        with netCDF4.Dataset(slot_spec['path']) as source:
            if int(getattr(source, 'run_complete', 0)) != 1:
                raise ValueError(f"Incomplete EF input: {slot_spec['path']}")
            if float(getattr(source, 'EF_filter_T_days', -1)) != expected_t:
                raise ValueError(
                    f"{slot_spec['path']} was written with T="
                    f"{getattr(source, 'EF_filter_T_days', 'unknown')}; the calibrated "
                    f"{slot_data} T is {expected_t:g}. Rerun 02_EF_RZSM_estimation.py --data {slot_data}."
                )
        ef_t_days[slot_data] = expected_t
    slot3_spec = land_product(DATA_ROOT, model3)

    RZSM_slot1, nc_lat_slot1, nc_lon_slot1, time_slot1 = load_rzsm_region(
        slot1_spec['path'], slot1_spec['variable'],
        ('lat', 'latitude'), ('lon', 'longitude'), slot1_spec['label']
    )
    RZSM_slot2, nc_lat_slot2, nc_lon_slot2, time_slot2 = load_rzsm_region(
        slot2_spec['path'], slot2_spec['variable'],
        ('lat', 'latitude'), ('lon', 'longitude'), slot2_spec['label']
    )
    RZSM_slot3, nc_lat_slot3, nc_lon_slot3, time_slot3 = load_rzsm_region(
        slot3_spec['path'], slot3_spec['variable'],
        ('lat', 'latitude'), ('lon', 'longitude'), slot3_spec['label']
    )
    slot1_architecture = (
        estimation_metadata(slot1_spec['path'], model1)
        if model1 in {'FNO', 'LSTM'} else None
    )
    slot2_architecture = (
        estimation_metadata(slot2_spec['path'], model2)
        if model2 in {'FNO', 'LSTM'} else None
    )

    RZSM_slot2, nc_lat_slot2, nc_lon_slot2 = align_or_regrid_to_reference(
        RZSM_slot2,
        nc_lat_slot1,
        nc_lon_slot1,
        nc_lat_slot2,
        nc_lon_slot2,
        slot2_spec['label'],
        allow_regrid=bool(slot2_spec['allow_regrid']),
    )
    RZSM_slot3, nc_lat_slot3, nc_lon_slot3 = align_or_regrid_to_reference(
        RZSM_slot3,
        nc_lat_slot1,
        nc_lon_slot1,
        nc_lat_slot3,
        nc_lon_slot3,
        slot3_spec['label'],
        allow_regrid=bool(slot3_spec['allow_regrid']),
    )

    (
        RZSM_slot1,
        RZSM_slot2,
        RZSM_slot3,
        aligned_time,
        anomaly_trim,
        time_metadata,
    ) = align_triplet_test_period(
        RZSM_slot1, time_slot1,
        RZSM_slot2, time_slot2,
        RZSM_slot3, time_slot3,
        SPINUP_DAYS,
        args.test_start_ratio,
        args.anomaly_window_days,
        args.test_end_date,
    )

    m, n, _ = RZSM_slot1.shape
    t = len(aligned_time)
    bootN = args.bootstrap_count
    sampleN = int(np.ceil(t * args.bootstrap_fraction))
    if sampleN <= nod_th:
        raise ValueError(
            f"Bootstrap sampleN={sampleN} is too small for nod_th={nod_th}. "
            "Use a longer test period or a smaller nod_th."
        )

    save_path = os.path.join(
        explicit_run_analysis_root(
            DATA_ROOT, 'test', args.run_name, args.period_folder
        ),
        'Raw',
    )
    os.makedirs(save_path, exist_ok=True)
    nc_file_name = os.path.join(
        save_path,
        explicit_tca_filename(model1, model2, model3),
    )

    spatial_chunks = make_spatial_chunks(m, n, chunk_m=args.chunk_m, chunk_n=args.chunk_n)
    bootstrap_indices = build_bootstrap_indices(
        t,
        bootN,
        sampleN,
        args.random_seed,
        method=args.bootstrap_method,
        block_length=args.block_length,
    )
    variable_names = (
        f"{slot1_spec['stem']}_VAR_err",
        f"{slot2_spec['stem']}_VAR_err",
        f"{slot3_spec['stem']}_VAR_err",
        f"{slot1_spec['stem']}_SNR",
        f"{slot2_spec['stem']}_SNR",
        f"{slot3_spec['stem']}_SNR",
        'flags_corr',
        'flags_n_valid',
        'flags_fMSE',
        'flags_negative_vars_err',
    )
    neural_static_schema = ''
    if model2 in ('FNO', 'LSTM'):
        with netCDF4.Dataset(slot2_spec['path']) as estimation_ds:
            neural_static_schema = getattr(estimation_ds, 'static_schema_version', '')
    attrs = {
        'neural_static_schema_version': neural_static_schema,
        'pipeline_version': PIPELINE_VERSION,
        'ef_estimation_root': os.path.abspath(args.ef_estimation_root),
        'neural_estimation_root': os.path.abspath(args.neural_estimation_root),
        'run_name': args.run_name or 'legacy_default',
        'ASCAT_EF_filter_T_days': ef_t_days.get('ASCAT', -1),
        'SMAP_EF_filter_T_days': ef_t_days.get('SMAP', -1),
        'model1': model1,
        'model2': model2,
        'model3': model3,
        'slot1_dataset': 'ASCAT',
        'slot2_dataset': 'SMAP',
        'slot2_substituted': int(args.slot2_file is not None),
        'triplet_product_1': slot1_spec['label'],
        'triplet_product_2': slot2_spec['label'],
        'triplet_product_3': slot3_spec['label'],
        'slot1_product_file': slot1_spec['path'],
        'slot2_product_file': slot2_spec['path'],
        'slot3_product_file': slot3_spec['path'],
        'input_fingerprints_json': fingerprints_json(
            (slot1_spec['path'], slot2_spec['path'], slot3_spec['path'])
        ),
        'slot3_regridded_to_estimation_grid': int(slot3_spec['allow_regrid']),
        'slot3_regridding_method': (
            'valid-neighbor bilinear interpolation'
            if slot3_spec['allow_regrid'] else 'none; coordinate alignment only'
        ),
        'analysis_period': 'test_period_after_ef_spinup',
        'anomaly_window_days': int(args.anomaly_window_days),
        'anomaly_definition': (
            'centered running nanmean of width anomaly_window_days subtracted '
            'from each triplet member before alignment'
        ),
        'spinup_days_considered': SPINUP_DAYS,
        'train_ratio': float(TRAIN_RATIO),
        'val_ratio': float(VAL_RATIO),
        'test_start_ratio_after_spinup': float(args.test_start_ratio),
        'test_period_policy': TEST_PERIOD_POLICY,
        'test_time_start': str(aligned_time[0]),
        'test_time_end': str(aligned_time[-1]),
        'test_time_steps': int(len(aligned_time)),
        'bootstrap_count': int(bootN),
        'bootstrap_sample_size': int(sampleN),
        'bootstrap_fraction': float(args.bootstrap_fraction),
        'bootstrap_method': args.bootstrap_method,
        'moving_block_length': (
            int(args.block_length) if args.bootstrap_method == 'moving_block' else 0
        ),
        'moving_block_length_rationale': (
            '30-day default approximately matches the 31-day anomaly window; '
            'it is not an empirical estimate of a decorrelation timescale, and '
            'the CLI value is configurable for sensitivity analysis.'
            if args.bootstrap_method == 'moving_block' else 'not applicable'
        ),
        'scientific_role': 'stability_QC_only_not_primary_central_estimate',
        'primary_central_estimator': 'deterministic held-out/full-data TCA from 02_TCA_single_run.py',
        'bootstrap_random_seed': int(args.random_seed),
        'bootstrap_rng': 'numpy.random.default_rng',
        'bootstrap_indices_sha256': bootstrap_indices_sha256(bootstrap_indices),
        'bootstrap_batch_size': int(args.bootstrap_batch_size),
        'result_storage': 'streamed atomic NetCDF4 writes',
        'grid_orientation': grid_orientation(nc_lat_slot1, nc_lon_slot1),
        f'ASCAT_{model1}_model': slot1_architecture,
        f'SMAP_{model2}_model': slot2_architecture,
        **time_metadata,
    }

    if args.support_output is not None:
        write_triplet_support(
            args.support_output,
            (RZSM_slot1, RZSM_slot2, RZSM_slot3),
            anomaly_trim,
            aligned_time,
            nc_lat_slot1,
            nc_lon_slot1,
            attrs,
            nod_th,
        )
        return

    temporary_output = os.path.join(
        save_path,
        f'.{os.path.basename(nc_file_name)}.tmp.{os.getpid()}',
    )
    if os.path.exists(temporary_output):
        os.remove(temporary_output)

    dataset = None
    try:
        with tempfile.TemporaryDirectory(prefix='tca_bootstrap_test_') as temp_dir:
            anomaly_paths = []
            slot_arrays = [RZSM_slot1, RZSM_slot2, RZSM_slot3]
            del RZSM_slot1, RZSM_slot2, RZSM_slot3
            for index, values in enumerate(slot_arrays, start=1):
                anomaly_path = os.path.join(temp_dir, f'anomaly_slot{index}.npy')
                context_path = os.path.join(
                    temp_dir, f'anomaly_slot{index}_context.npy'
                )
                write_rolling_anomaly(
                    values,
                    context_path,
                    args.anomaly_window_days,
                    args.chunk_m,
                    args.chunk_n,
                )
                context_anomaly = np.load(context_path, mmap_mode='r')
                save_npy_atomic(
                    anomaly_path,
                    np.ascontiguousarray(context_anomaly[:, :, anomaly_trim]),
                )
                del context_anomaly
                os.remove(context_path)
                anomaly_paths.append(anomaly_path)
                slot_arrays[index - 1] = None
                gc.collect()
            del values
            del slot_arrays
            gc.collect()

            indices_path = os.path.join(temp_dir, 'bootstrap_indices.npy')
            save_npy_atomic(indices_path, bootstrap_indices)
            dataset = create_raw_dataset(
                temporary_output,
                nc_lat_slot1,
                nc_lon_slot1,
                bootstrap_indices,
                variable_names,
                attrs,
                args.chunk_m,
                args.chunk_n,
                args.bootstrap_batch_size,
            )
            total_tasks = task_count(
                len(spatial_chunks),
                bootN,
                args.bootstrap_batch_size,
            )
            context = mp.get_context('spawn')
            with context.Pool(
                processes=min(args.n_cores, mp.cpu_count()),
                initializer=initialize_worker,
                initargs=(
                    tuple(anomaly_paths),
                    indices_path,
                    nod_th,
                    corr_th,
                ),
            ) as pool:
                results = pool.imap_unordered(
                    process_bootstrap_batch,
                    iter_bootstrap_tasks(
                        spatial_chunks,
                        bootN,
                        args.bootstrap_batch_size,
                    ),
                    chunksize=1,
                )
                for result in tqdm(
                    results,
                    total=total_tasks,
                    desc=os.environ.get('RZSM_TCA_PROGRESS_DESC',
                                        f'{model1} - {model2} - {model3}'),
                    leave=False,
                    dynamic_ncols=True,
                    smoothing=0.03,
                    mininterval=2.0,
                    bar_format=(
                        '{desc}: {percentage:.0f}% | {n_fmt}/{total_fmt} '
                        '[{elapsed}<{remaining}, {rate_fmt}]'
                    ),
                ):
                    write_bootstrap_batch(dataset, variable_names, result)

            dataset.run_complete = 1
            dataset.sync()
            dataset.close()
            dataset = None
        os.replace(temporary_output, nc_file_name)
    except BaseException:
        if dataset is not None:
            dataset.close()
        if os.path.exists(temporary_output):
            os.remove(temporary_output)
        raise


def masking_main(argv=None):
    george_FP = str(default_george_fp())

    parser = argparse.ArgumentParser(
        description="Mask one explicit test-period TCA triplet using quality flags."
    )
    parser.add_argument('--model1', required=True, choices=EXPLICIT_MODELS)
    parser.add_argument('--model2', required=True, choices=EXPLICIT_MODELS)
    parser.add_argument('--model3', required=True, choices=EXPLICIT_LAND_PRODUCTS)
    parser.add_argument('--run_name', default=None)
    parser.add_argument('--period_folder', default=None)
    args = parser.parse_args(argv)
    require_fixed_ef_model_pair(args.model1, args.model2)

    TCA_ROOT = explicit_run_analysis_root(
        george_FP, 'test', args.run_name, args.period_folder
    )
    RAW_DIR = os.path.join(TCA_ROOT, 'Raw')
    MASKED_DIR = os.path.join(TCA_ROOT, 'Masked')
    os.makedirs(MASKED_DIR, exist_ok=True)

    filename = explicit_tca_filename(args.model1, args.model2, args.model3)
    raw_path = os.path.join(RAW_DIR, filename)
    masked_path = os.path.join(MASKED_DIR, filename)

    if not os.path.isfile(raw_path):
        raise FileNotFoundError(f"Raw test-period TCA file not found: {raw_path}")

    temporary_path = f'{masked_path}.tmp.{os.getpid()}'
    if os.path.exists(temporary_path):
        os.remove(temporary_path)

    try:
        with netCDF4.Dataset(raw_path) as source:
            if int(getattr(source, 'run_complete', 0)) != 1:
                raise RuntimeError(
                    f"Raw test-period TCA file is incomplete: {raw_path}"
                )
            variables_to_mask = [
                name for name in source.variables
                if 'VAR_err' in name or 'SNR' in name
            ]
            m, n, boot_n = (len(source.dimensions[name])
                            for name in ('lat', 'lon', 'boot'))
            first_chunks = source[variables_to_mask[0]].chunking()
            chunk_m, chunk_n, chunk_b = (
                tuple(first_chunks) if first_chunks != 'contiguous' else (20, 20, 25)
            )
            spatial_blocks = [
                (slice(i, min(i + chunk_m, m)), slice(j, min(j + chunk_n, n)))
                for i in range(0, m, chunk_m)
                for j in range(0, n, chunk_n)
            ]
            boot_blocks = [slice(k, min(k + chunk_b, boot_n))
                           for k in range(0, boot_n, chunk_b)]

            with netCDF4.Dataset(temporary_path, 'w', format='NETCDF4') as masked:
                for dim in ('lat', 'lon', 'boot'):
                    masked.createDimension(dim, len(source.dimensions[dim]))
                for coord in ('lat', 'lon'):
                    output = masked.createVariable(coord, source[coord].dtype, (coord,))
                    output[:] = source[coord][:]
                    output.setncatts({key: source[coord].getncattr(key)
                                      for key in source[coord].ncattrs()})
                output_variables = {}
                for input_name in variables_to_mask:
                    output_name = (input_name.replace('VAR_err', 'err')
                                   if 'VAR_err' in input_name
                                   else input_name.replace('SNR', 'R'))
                    output_variables[input_name] = masked.createVariable(
                        output_name, 'f4', ('lat', 'lon', 'boot'),
                        fill_value=np.nan, zlib=True, complevel=4, shuffle=True,
                        chunksizes=(chunk_m, chunk_n, chunk_b),
                    )

                total = len(spatial_blocks) * len(boot_blocks)
                with tqdm(total=total, desc=os.environ.get('RZSM_TCA_PROGRESS_DESC',
                                              f'Masking {args.model1}-{args.model2}-{args.model3}'),
                          dynamic_ncols=True, smoothing=0.03, mininterval=2.0,
                          leave=False) as bar:
                    for lat_slice, lon_slice in spatial_blocks:
                        for boot_slice in boot_blocks:
                            index = (lat_slice, lon_slice, boot_slice)
                            valid = np.ones(
                                source['flags_corr'][index].shape, dtype=bool
                            )
                            for flag in ('flags_corr', 'flags_n_valid', 'flags_fMSE',
                                         'flags_negative_vars_err'):
                                valid &= np.ma.filled(source[flag][index], 1) == 0
                            for input_name, output in output_variables.items():
                                values = np.ma.filled(source[input_name][index], np.nan)
                                values = np.where(valid, values, np.nan)
                                with np.errstate(invalid='ignore', divide='ignore'):
                                    transformed = (np.sqrt(values) if 'VAR_err' in input_name
                                                   else np.sqrt(1 / (1 + 1 / values)))
                                output[index] = transformed.astype(np.float32)
                            bar.update()

                masked.setncatts({key: source.getncattr(key)
                                  for key in source.ncattrs()})
                masked.processing_stage = 'Masked'
                masked.run_complete = 1
                masked.mask_source_file = raw_path
                masked.R_definition = 'sqrt(SNR / (1 + SNR)); TCA correlation coefficient'
        os.replace(temporary_path, masked_path)
    except BaseException:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)
        raise


def averaging_main(argv=None):
    george_FP = str(default_george_fp())

    parser = argparse.ArgumentParser(
        description="Average one explicit test-period masked TCA triplet."
    )
    parser.add_argument('--model1', required=True, choices=EXPLICIT_MODELS)
    parser.add_argument('--model2', required=True, choices=EXPLICIT_MODELS)
    parser.add_argument('--model3', required=True, choices=EXPLICIT_LAND_PRODUCTS)
    parser.add_argument('--run_name', default=None)
    parser.add_argument('--period_folder', default=None)
    args = parser.parse_args(argv)
    require_fixed_ef_model_pair(args.model1, args.model2)

    TCA_ROOT = explicit_run_analysis_root(
        george_FP, 'test', args.run_name, args.period_folder
    )
    MASKED_DIR = os.path.join(TCA_ROOT, 'Masked')
    MERGED_DIR = os.path.join(TCA_ROOT, 'Merged')
    os.makedirs(MERGED_DIR, exist_ok=True)

    filename = explicit_tca_filename(args.model1, args.model2, args.model3)
    masked_path = os.path.join(MASKED_DIR, filename)
    merged_path = os.path.join(MERGED_DIR, filename)

    if not os.path.isfile(masked_path):
        raise FileNotFoundError(
            f"Masked test-period TCA file not found: {masked_path}"
        )

    temporary_path = f'{merged_path}.tmp.{os.getpid()}'
    if os.path.exists(temporary_path):
        os.remove(temporary_path)
    try:
        with netCDF4.Dataset(masked_path) as source:
            if int(getattr(source, 'run_complete', 0)) != 1:
                raise RuntimeError(
                    f"Masked test-period TCA file is incomplete: {masked_path}"
                )
            variables = [name for name, variable in source.variables.items()
                         if variable.dimensions == ('lat', 'lon', 'boot')]
            m, n = len(source.dimensions['lat']), len(source.dimensions['lon'])
            first_chunks = source[variables[0]].chunking()
            chunk_m, chunk_n = (
                tuple(first_chunks[:2]) if first_chunks != 'contiguous' else (20, 20)
            )
            blocks = [
                (slice(i, min(i + chunk_m, m)), slice(j, min(j + chunk_n, n)))
                for i in range(0, m, chunk_m)
                for j in range(0, n, chunk_n)
            ]
            with netCDF4.Dataset(temporary_path, 'w', format='NETCDF4') as merged:
                for dim in ('lat', 'lon'):
                    merged.createDimension(dim, len(source.dimensions[dim]))
                    coord = merged.createVariable(dim, source[dim].dtype, (dim,))
                    coord[:] = source[dim][:]
                    coord.setncatts({key: source[dim].getncattr(key)
                                     for key in source[dim].ncattrs()})
                outputs = {
                    name: merged.createVariable(
                        name, 'f4', ('lat', 'lon'), fill_value=np.nan,
                        zlib=True, complevel=4, shuffle=True,
                        chunksizes=(chunk_m, chunk_n),
                    ) for name in variables
                }
                total = len(blocks) * len(variables)
                with tqdm(total=total, desc=os.environ.get('RZSM_TCA_PROGRESS_DESC',
                                              f'Averaging {args.model1}-{args.model2}-{args.model3}'),
                          dynamic_ncols=True, smoothing=0.03, mininterval=2.0,
                          leave=False) as bar:
                    for lat_slice, lon_slice in blocks:
                        for name, output in outputs.items():
                            values = np.ma.filled(
                                source[name][lat_slice, lon_slice, :], np.nan
                            ).astype(np.float64)
                            with warnings.catch_warnings():
                                warnings.simplefilter('ignore', RuntimeWarning)
                                output[lat_slice, lon_slice] = np.nanmean(values, axis=2)
                            bar.update()
                merged.setncatts({key: source.getncattr(key)
                                  for key in source.ncattrs()})
                merged.processing_stage = 'Merged'
                merged.run_complete = 1
                merged.boot_average_source_file = masked_path
                merged.bootstrap_aggregation = 'arithmetic mean over valid bootstraps'
                merged.scientific_role = 'stability_QC_only_not_primary_central_estimate'
                merged.not_primary_central_estimate = 1
                merged.primary_central_estimator = (
                    'deterministic held-out-period TCA from 02_TCA_single_run.py --type test'
                )
        os.replace(temporary_path, merged_path)
    except BaseException:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)
        raise


def generate_lon_lat_axes(*args, bounds=None):
    '''Generate longitude and latitude axes for an equal-angle grid.'''
    if len(args) == 1:
        resolution = float(args[0])
        if not np.isfinite(resolution) or resolution <= 0:
            raise ValueError('Grid resolution must be finite and positive.')
        rows, columns = int(180 / resolution), int(360 / resolution)
    elif len(args) == 2:
        rows, columns = args
        if not all(np.isfinite(value) and int(value) == value for value in (rows, columns)):
            raise ValueError('Grid dimensions must be finite integers.')
        rows, columns = int(rows), int(columns)
    else:
        raise ValueError('Provide a resolution or the grid row and column counts.')
    if rows < 1 or columns < 1:
        raise ValueError('Grid dimensions must be positive.')
    latitude = np.linspace(90 - 90 / rows, -90 + 90 / rows, rows)
    longitude = np.linspace(-180 + 180 / columns, 180 - 180 / columns, columns)
    if bounds is not None and len(bounds):
        if len(bounds) != 4 or not np.isfinite(bounds).all():
            raise ValueError('Bounds must contain finite west, east, south and north values.')
        west, east, south, north = bounds
        if not west < east or not south < north:
            raise ValueError('Bounds must increase from west to east and south to north.')
        longitude = longitude[(longitude > west) & (longitude < east)]
        latitude = latitude[(latitude > south) & (latitude < north)]
        if not len(longitude) or not len(latitude):
            raise ValueError('No grid cell centres fall inside the requested bounds.')
    return longitude, latitude


def generate_lon_lat_eqdgrid(*args, bounds=None):
    '''Generate longitude and latitude meshes in grid index order.'''
    longitude, latitude = generate_lon_lat_axes(*args, bounds=bounds)
    return np.meshgrid(longitude, latitude)


def nearest_indices(target_values, source_values):
    return np.array(
        [int(np.argmin(np.abs(target_values - value))) for value in source_values],
        dtype=np.int64
    )


def contiguous_slice(indices):
    if len(indices) == 0:
        raise ValueError("Cannot build a slice from empty indices.")

    indices = np.asarray(indices, dtype=np.int64)

    if np.all(np.diff(indices) == 1):
        return slice(int(indices[0]), int(indices[-1]) + 1), False

    if np.all(np.diff(indices) == -1):
        return slice(int(indices[-1]), int(indices[0]) + 1), True

    return None, False


def copy_variable_attrs(source_var, target_var, skip_attrs=('_FillValue',)):
    for attr_name in source_var.ncattrs():
        if attr_name in skip_attrs:
            continue
        setattr(target_var, attr_name, getattr(source_var, attr_name))


def copy_global_attrs(source_ds, target_ds, extra_attrs=None):
    for attr_name in source_ds.ncattrs():
        setattr(target_ds, attr_name, getattr(source_ds, attr_name))

    if extra_attrs is not None:
        for attr_name, attr_value in extra_attrs.items():
            setattr(target_ds, attr_name, attr_value)


def get_global_eqd_lat_lon():
    global_lon_2d, global_lat_2d = generate_lon_lat_eqdgrid(0.10)
    global_lat = global_lat_2d[:, 0].astype(np.float32)
    global_lon = global_lon_2d[0, :].astype(np.float32)
    return global_lat, global_lon


def reorder_by_dim(data, dims, flip_lat=False, flip_lon=False):
    if flip_lat and 'lat' in dims:
        lat_axis = dims.index('lat')
        data = np.flip(data, axis=lat_axis)

    if flip_lon and 'lon' in dims:
        lon_axis = dims.index('lon')
        data = np.flip(data, axis=lon_axis)

    return data


def clean_tca(values):
    arr = np.ma.filled(values, np.nan).astype(np.float32)
    return arr


def expand_tca_file(input_file, output_file):
    input_file = Path(input_file)
    output_file = Path(output_file)
    global_lat, global_lon = get_global_eqd_lat_lon()

    with netCDF4.Dataset(str(input_file), 'r') as in_ds:
        local_lat = np.asarray(in_ds['lat'][:], dtype=np.float32)
        local_lon = np.asarray(in_ds['lon'][:], dtype=np.float32)

        lat_indices = nearest_indices(global_lat, local_lat)
        lon_indices = nearest_indices(global_lon, local_lon)

        lat_slice, flip_lat = contiguous_slice(lat_indices)
        lon_slice, flip_lon = contiguous_slice(lon_indices)

        if lat_slice is None or lon_slice is None:
            raise ValueError(
                "Input latitude/longitude indices are not contiguous on the global output grid.\n"
                f"lat index sample: {lat_indices[:5]} ... {lat_indices[-5:]}\n"
                f"lon index sample: {lon_indices[:5]} ... {lon_indices[-5:]}"
            )

        output_file.parent.mkdir(parents=True, exist_ok=True)

        with netCDF4.Dataset(str(output_file), 'w', format='NETCDF4') as out_ds:
            for dim_name, dim in in_ds.dimensions.items():
                if dim_name == 'lat':
                    out_ds.createDimension('lat', len(global_lat))
                elif dim_name == 'lon':
                    out_ds.createDimension('lon', len(global_lon))
                else:
                    out_ds.createDimension(
                        dim_name,
                        None if dim.isunlimited() else len(dim)
                    )

            copy_global_attrs(in_ds, out_ds, {
                'expand_source': os.path.basename(input_file),
                'expand_grid': '0.1_degree_equal_grid_global',
                'expand_note': (
                    'Original regional TCA values were inserted into the matching global grid cells; '
                    'all other cells remain NaN.'
                ),
                'expand_flip_lat': str(flip_lat),
                'expand_flip_lon': str(flip_lon)
            })

            for var_name, source_var in in_ds.variables.items():
                dims = source_var.dimensions

                if var_name == 'lat':
                    target_var = out_ds.createVariable('lat', 'f4', ('lat',))
                    target_var[:] = global_lat
                    copy_variable_attrs(source_var, target_var)
                    continue

                if var_name == 'lon':
                    target_var = out_ds.createVariable('lon', 'f4', ('lon',))
                    target_var[:] = global_lon
                    copy_variable_attrs(source_var, target_var)
                    continue

                fill_value = getattr(source_var, '_FillValue', None)

                if 'lat' in dims and 'lon' in dims:
                    target_var = out_ds.createVariable(
                        var_name,
                        'f4',
                        dims,
                        zlib=True,
                        complevel=4,
                        fill_value=np.nan
                    )
                    copy_variable_attrs(source_var, target_var)

                    insert_slices = []

                    for dim_name in dims:
                        if dim_name == 'lat':
                            insert_slices.append(lat_slice)
                        elif dim_name == 'lon':
                            insert_slices.append(lon_slice)
                        else:
                            insert_slices.append(slice(None))

                    local_data = clean_tca(source_var[:])
                    local_data = reorder_by_dim(
                        local_data,
                        dims,
                        flip_lat=flip_lat,
                        flip_lon=flip_lon
                    )

                    target_var[tuple(insert_slices)] = local_data

                else:
                    if fill_value is None:
                        target_var = out_ds.createVariable(
                            var_name,
                            source_var.datatype,
                            dims
                        )
                    else:
                        target_var = out_ds.createVariable(
                            var_name,
                            source_var.datatype,
                            dims,
                            fill_value=fill_value
                        )

                    copy_variable_attrs(source_var, target_var)
                    target_var[:] = source_var[:]

    print(f"Expanded file saved: {output_file}")
    print(f"Inserted lat slice: {lat_slice.start}:{lat_slice.stop}, flip_lat={flip_lat}")
    print(f"Inserted lon slice: {lon_slice.start}:{lon_slice.stop}, flip_lon={flip_lon}")
    return output_file


def tca_directory_name(option, add, scheme):
    if add is None:
        return analysis_folder(option, scheme)
    if option != 'default':
        raise ValueError('--add SMAP_L4 cannot be combined with a non-default --option.')
    if add == 'SMAP_L4':
        return os.path.join(analysis_folder(option, scheme), 'SMAP_L4')
    raise ValueError(
        f"Unsupported TCA option/add/scheme combination: {option}, {add}, {scheme}"
    )


def expand_tca_results(args):
    tca_FP = os.path.join(project_results_root(DATA_ROOT), 'CONUS_RZSM_Estimation', tca_directory_name(args.option, args.add, args.scheme))

    input_FP = os.path.join(tca_FP, 'Merged')
    output_FP = os.path.join(tca_FP, 'Expanded')

    for model_arg in args.model:
        for current_data in args.data:
            input_file = args.input_file
            if input_file is None:
                input_file = os.path.join(
                    input_FP,
                    f'TCA_{model_arg}_{current_data}.nc'
                )

            output_file = args.output_file
            if output_file is None:
                output_file = os.path.join(
                    output_FP,
                    f'TCA_{model_arg}_{current_data}_expanded.nc'
                )

            if not os.path.exists(input_file):
                print(f"Input file not found: {input_file}")
                continue

            expand_tca_file(input_file, output_file)


def parse_expansion_args(argv=None):
    parser = argparse.ArgumentParser(description="Expand merged TCA results to the global equal grid")
    parser.add_argument(
        '--model',
        nargs='+',
        type=str,
        default=['EF', 'LSTM', 'FNO'],
        choices=SUPPORTED_MODELS,
        help="Model names: EF, FNO, or LSTM.",
    )
    parser.add_argument(
        '--data',
        nargs='+',
        type=str,
        default=['ASCAT', 'SMAP'],
        choices=SUPPORTED_DATA,
    )
    parser.add_argument('--scheme', type=str, default='full', choices=['full', 'test'])
    parser.add_argument(
        '--option',
        type=str,
        default='default',
        choices=TCA_OPTIONS,
        help=(
            'TCA triplet option: default uses EF(other sensor), amsr2 uses '
            'EF(AMSR2 nighttime), and nldas uses NLDAS(0-40 cm).'
        ),
    )
    parser.add_argument(
        '--add',
        type=str,
        default=None,
        choices=SUPPORTED_ADD,
        help="Optional added middle product. Currently supported: SMAP_L4.",
    )
    parser.add_argument('--input_file', type=str, default=None, help='Optional direct input file path')
    parser.add_argument('--output_file', type=str, default=None, help='Optional direct output file path')
    args = parser.parse_args(argv)

    if args.add is not None and args.option != 'default':
        raise ValueError('--add SMAP_L4 cannot be combined with a non-default --option.')
    if args.add == 'SMAP_L4' and any(data != 'ASCAT' for data in args.data):
        raise ValueError("--add SMAP_L4 only supports --data ASCAT.")

    return args


def expansion_main(argv=None):
    args = parse_expansion_args(argv)
    expand_tca_results(args)
    add_label = args.add if args.add is not None else 'none'
    print(
        f"05_TCA_calculation.py expand finished for {args.model} model with {args.data} data "
        f"using option={args.option}, add={add_label}, scheme={args.scheme}."
    )


def normalize_model3(value):
    value = value.strip().upper().replace("_", "-")
    aliases = {
        "ERA5": "ERA5-Land", "ERA5-LAND": "ERA5-Land",
        "NOAH": "NOAH", "NLDAS-NOAH": "NOAH",
        "VIC": "VIC", "NLDAS-VIC": "VIC",
        "MOS": "MOS", "MOSAIC": "MOS",
        "NLDAS-MOS": "MOS", "NLDAS-MOSAIC": "MOS",
    }
    if value not in aliases:
        raise argparse.ArgumentTypeError(
            "--model must be ERA5, NOAH, VIC, or MOSAIC."
        )
    return aliases[value]


def test_period_folder(anomaly, test_end_date):
    '''Identify the bootstrap folder for one anomaly window and evaluation end.

    The default configuration - the default anomaly window over the full
    available test period - is the run the manuscript reads, so it keeps the
    canonical ``Triple_Collocation_Analysis`` folder. Anything else is a
    sensitivity run and is routed to its own ``Test_period_<window>d`` folder
    so it can never overwrite the canonical one.
    '''
    if anomaly == ANOMALY_WINDOW_DAYS and test_end_date is None:
        return TCA_PERIOD_FOLDERS['test']
    return f'Test_period_{anomaly}d'


def triplet_args(
    model2,
    model3,
    *,
    workers=DEFAULT_WORKERS,
    random_seed=DEFAULT_RANDOM_SEED,
    anomaly=ANOMALY_WINDOW_DAYS,
    test_end_date=None,
):
    '''Build the settings for one TCA triplet.'''
    return argparse.Namespace(
        model1='EF',
        model2=model2,
        model3=model3,
        workers=workers,
        random_seed=random_seed,
        anomaly_window_days=anomaly,
        test_end_date=test_end_date,
        test_period_folder=test_period_folder(anomaly, test_end_date),
        run_name=None,
        neural_estimation_root=default_neural_estimation_root(),
    )


def run_command(command, log=None, desc=None):
    '''Run one stage; stdout goes to ``log``, stderr (its progress bar) to the terminal.'''
    env = None if desc is None else {**os.environ, PROGRESS_DESC_ENV: desc}
    subprocess.run(command, cwd=SCRIPT_DIR, check=True, stdout=log, env=env)


def common_triplet_arguments(args):
    command = [
        '--model1', args.model1,
        '--model2', args.model2,
        '--model3', args.model3,
    ]
    if args.run_name is not None:
        command.extend(['--run_name', args.run_name])
    command.extend(['--period_folder', args.test_period_folder])
    return command


def calculation_command(args):
    command = [
        sys.executable,
        str(SCRIPT_PATH),
        'bootstrap',
        *common_triplet_arguments(args),
        '--n_cores', str(args.workers),
        '--random_seed', str(args.random_seed),
        '--anomaly_window_days', str(args.anomaly_window_days),
        '--neural_estimation_root', str(args.neural_estimation_root),
    ]
    if args.test_end_date is not None:
        command.extend(['--test_end_date', args.test_end_date])
    return command


def list_stage_command(stage, args):
    return [
        sys.executable,
        str(SCRIPT_PATH),
        {'masking': 'mask', 'averaging': 'average'}[stage],
        *common_triplet_arguments(args),
    ]


def validate_workflow_args(args):
    if args.model1 not in EXPLICIT_MODELS:
        raise ValueError(f"Unsupported --model1: {args.model1!r}")
    if args.model2 not in EXPLICIT_MODELS:
        raise ValueError(f"Unsupported --model2: {args.model2!r}")
    if args.model3 not in EXPLICIT_LAND_PRODUCTS:
        raise ValueError(f"Unsupported --model3: {args.model3!r}")
    require_fixed_ef_model_pair(args.model1, args.model2)


def expand_bootstrap_result(args):
    root = Path(explicit_run_analysis_root(
        str(default_george_fp()),
        'test',
        args.run_name,
        args.test_period_folder,
    ))
    filename = explicit_tca_filename(args.model1, args.model2, args.model3)
    input_path = root / 'Merged' / filename
    output_path = root / 'Expanded' / f'{Path(filename).stem}_expanded.nc'

    if not input_path.is_file():
        raise FileNotFoundError(
            f'Merged TCA result is unavailable for expansion: {input_path}'
        )
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        expand_tca_file(input_path, output_path)
    return output_path


def run_tca_workflow(args, log=None, label=None):
    '''Run every TCA stage for one triplet; ``log`` receives the stage stdout.'''
    validate_workflow_args(args)
    ensure_tca_layout(default_george_fp())
    for stage in TCA_STAGES:
        if stage == 'expansion':
            expanded_path = expand_bootstrap_result(args)
            continue
        command = (
            calculation_command(args)
            if stage == 'calculation'
            else list_stage_command(stage, args)
        )
        run_command(command, log, desc=None if label is None else f'{label} {stage}')

    return {
        'triplet': explicit_triplet_label(args.model1, args.model2, args.model3),
        'filename': explicit_tca_filename(args.model1, args.model2, args.model3),
        'stages': list(TCA_STAGES),
        'expanded': expanded_path,
    }


def triplet_log_path(args):
    '''Identify the log file that receives one triplet's stage stdout.'''
    root = Path(explicit_run_analysis_root(
        str(default_george_fp()), 'test', args.run_name, args.test_period_folder,
    ))
    filename = explicit_tca_filename(args.model1, args.model2, args.model3)
    return root / 'Logs' / f'{Path(filename).stem}.log'


def run_triplets(jobs):
    '''Run triplets sequentially with per-stage progress bars and logs.'''
    results = []
    for label, args in jobs:
        log_path = triplet_log_path(args)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open('w') as log:
            try:
                results.append(run_tca_workflow(args, log=log, label=label))
            except subprocess.CalledProcessError as error:
                raise TripletFailed(
                    f'{label} failed with exit code {error.returncode}'
                ) from error
    return results


class TripletFailed(RuntimeError):
    pass


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Individual stages: bootstrap, mask, average, expand.",
    )
    parser.add_argument(
        "--rzsm", type=str.upper, choices=RZSM_MODELS,
        help="SMAP RZSM model (default: all three).",
    )
    parser.add_argument(
        "--model", type=normalize_model3, choices=LAND_PRODUCTS,
        metavar="{ERA5,NOAH,VIC,MOSAIC}",
        help="Land model (default: ERA5, NOAH, VIC, and MOSAIC).",
    )
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"Workers per TCA calculation (default: {DEFAULT_WORKERS}).")
    parser.add_argument("--random_seed", type=int, default=DEFAULT_RANDOM_SEED)
    args = parser.parse_args(argv)
    if args.workers <= 0:
        parser.error("--workers must be positive.")
    if args.random_seed < 0:
        parser.error("--random_seed must be non-negative.")
    return args


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    stages = {
        'bootstrap': bootstrap_main,
        'mask': masking_main,
        'average': averaging_main,
        'expand': expansion_main,
    }
    if argv and argv[0] in stages:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return stages[argv[0]](argv[1:])

    args = parse_args(argv)
    jobs = [
        (f"EF-{rzsm}-{model}", triplet_args(
            rzsm, model, workers=args.workers,
            random_seed=args.random_seed,
        ))
        for rzsm in ((args.rzsm,) if args.rzsm else RZSM_MODELS)
        for model in ((args.model,) if args.model else LAND_PRODUCTS)
    ]
    return run_triplets(jobs)


if __name__ == "__main__":
    try:
        main()
    except TripletFailed as error:
        sys.exit(f"Stopped: {error}")
    if sys.argv[1:2] not in (['bootstrap'], ['mask'], ['average'], ['expand']):
        print("05_TCA_calculation.py is finished.")
