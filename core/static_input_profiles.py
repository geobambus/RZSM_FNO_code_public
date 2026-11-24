"""The ordered static predictors used by the published neural models."""

import torch

from core.project_paths import existing_result_path, project_results_root

EVENT_FEATURES = ('NDVI_avg', 'Bulk_density_RZSM')
EVENT_SCHEMA = 'RZSM_ndvi_bulk_density_event_precipitation_n1_v1'
EVENT_PROFILE = 'ndvi_bulk_event_precipitation'
DEFAULT_PROFILE = EVENT_PROFILE
FAMILY_DIRECTORIES = {'FNO': 'Fourier_Neural_Operator', 'LSTM': 'Long_Short-Term_Memory'}


def features(profile):
    if profile != EVENT_PROFILE:
        raise ValueError(f'Unsupported published input profile {profile!r}; expected {EVENT_PROFILE!r}.')
    return list(EVENT_FEATURES)


def schema(profile):
    features(profile)
    return EVENT_SCHEMA


def grid_path(project, profile, *, read_existing=False):
    schema(profile)
    path = project_results_root(project) / 'CONUS_RZSM_Estimation/Model_input.nc'
    return existing_result_path(path) if read_existing else path


def validate_grid_schema(dataset, profile, path):
    expected = features(profile)
    if (getattr(dataset, 'static_schema_version', None) != schema(profile)
            or getattr(dataset, 'static_features', '').split(',') != expected
            or getattr(dataset, 'run_complete', 0) != 1):
        raise ValueError(f'{path}: grid schema/order is incompatible; rerun 01_Model_input.py.')
    missing = [name for name in expected if name not in dataset.variables]
    if missing:
        raise KeyError(f'{path}: missing static predictors {missing}')
    for name in expected:
        if dataset.variables[name].dimensions != ('lat', 'lon'):
            raise ValueError(f'{path}: {name} must be one static value per pixel (lat, lon), with no time dimension')


def augment_static_inputs(inputs, feature_cols, dropout_rate=.05, noise_std=.05):
    """Apply the published continuous-input augmentation in its original RNG order."""
    if tuple(feature_cols) != EVENT_FEATURES:
        raise ValueError(f'Static augmentation requires the published ordered features {EVENT_FEATURES}.')
    mask = (torch.rand_like(inputs) > dropout_rate).float()
    noise = torch.randn_like(inputs) * noise_std
    return inputs * mask + noise * mask
