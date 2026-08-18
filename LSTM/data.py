"""Point-series preparation and PyTorch datasets for LSTM training."""

import os
import random

import netCDF4 as nc
import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, Dataset

from .settings import (
    PRODUCT_SETTINGS,
    SPATIAL_SPLIT_SEED,
    SPATIAL_TRAIN_FRACTION,
    STATIC_FEATURES,
    TARGET_VARIABLE,
    WINDOW_SIZE,
    build_observation_sequences,
    read_point_series,
    split_boundaries,
    validate_product,
    validate_static_columns,
)


class SequenceDataset(Dataset):
    """Observation-event sequences for one point and one target interval."""

    def __init__(
        self,
        ssm,
        target,
        static_properties,
        dates,
        target_start,
        target_stop,
        pixel_id,
    ):
        dynamic, targets, target_indices = build_observation_sequences(
            ssm,
            target,
            time_values=dates,
            target_start=target_start,
            target_stop=target_stop,
            window_size=WINDOW_SIZE,
        )
        self.target_indices = target_indices
        self.dynamic = torch.tensor(dynamic, dtype=torch.float32)
        self.target = torch.tensor(targets, dtype=torch.float32)
        repeated_static = np.tile(static_properties, (len(dynamic), 1))
        self.static = torch.tensor(repeated_static, dtype=torch.float32)
        self.pixel_id = int(pixel_id)

    def __len__(self):
        return len(self.target)

    def __getitem__(self, index):
        return (
            self.dynamic[index].contiguous(),
            self.static[index].contiguous(),
            self.target[index],
            self.pixel_id,
        )


def spatial_training_indices(row_count):
    """Return deterministic 80% training and 20% spatial-validation rows."""
    indices = list(range(int(row_count)))
    random.Random(SPATIAL_SPLIT_SEED).shuffle(indices)
    boundary = int(len(indices) * SPATIAL_TRAIN_FRACTION)
    if boundary <= 0 or boundary >= len(indices):
        raise RuntimeError("At least two point rows are required for spatial splitting.")
    return indices[:boundary], indices[boundary:]


def compute_scaler(input_frame):
    """Fit feature means/SDs using only the spatial-training point rows."""
    feature_names = validate_static_columns(input_frame.columns)
    training_indices, _ = spatial_training_indices(len(input_frame))
    training_frame = input_frame.iloc[training_indices]
    mean = training_frame[feature_names].mean()
    standard_deviation = training_frame[feature_names].std()
    if not np.all(np.isfinite(mean.to_numpy(dtype=float))):
        raise ValueError("Static-feature means contain non-finite values.")
    if not np.all(np.isfinite(standard_deviation.to_numpy(dtype=float))):
        raise ValueError("Static-feature standard deviations contain non-finite values.")
    return {
        "feature_names": list(feature_names),
        "mean": [float(mean[name]) for name in feature_names],
        "standard_deviation": [
            float(standard_deviation[name]) for name in feature_names
        ],
    }


def apply_scaler(input_frame, scaler):
    """Return a copy with the bundle's ordered static features standardized."""
    feature_names = list(scaler["feature_names"])
    validate_static_columns(input_frame.columns)
    if feature_names != list(STATIC_FEATURES):
        raise ValueError("Scaler feature order does not match the point-model features.")
    scaled = input_frame.copy()
    for name, mean, standard_deviation in zip(
        feature_names,
        scaler["mean"],
        scaler["standard_deviation"],
    ):
        if np.isfinite(standard_deviation) and standard_deviation != 0:
            scaled[name] = (scaled[name] - mean) / standard_deviation
        else:
            scaled[name] = 0.0
    return scaled


def build_training_datasets(input_csv, ismn_root, product, scaler):
    """Build temporal and spatial point splits used by the active experiment."""
    product = validate_product(product)
    input_frame = pd.read_csv(input_csv)
    if input_frame.empty:
        raise RuntimeError(f"Training CSV is empty: {input_csv}")
    if input_frame.duplicated(["lat_idx", "lon_idx"]).any():
        raise RuntimeError(f"Training CSV contains duplicate pixels: {input_csv}")
    feature_names = validate_static_columns(input_frame.columns, input_csv)
    scaled_frame = apply_scaler(input_frame, scaler)
    training_indices, _ = spatial_training_indices(len(input_frame))
    training_indices = set(training_indices)

    training_datasets = []
    temporal_validation_datasets = []
    spatial_validation_datasets = []
    settings = PRODUCT_SETTINGS[product]

    for row_index, row in scaled_frame.iterrows():
        latitude_index = int(row["lat_idx"])
        longitude_index = int(row["lon_idx"])
        point_file = os.path.join(
            ismn_root,
            "ISMN_Train",
            settings["folder"],
            f"{latitude_index}_{longitude_index}.nc",
        )
        if not os.path.exists(point_file):
            raise FileNotFoundError(f"Missing training point file: {point_file}")
        with nc.Dataset(point_file) as point_dataset:
            ssm, target, dates, _ = read_point_series(
                point_dataset, product, TARGET_VARIABLE
            )
        target = np.asarray(target, dtype=float)
        target[~np.isfinite(ssm)] = np.nan
        train_stop, validation_stop = split_boundaries(dates)
        static_properties = row[feature_names].to_numpy(dtype=float)

        if row_index in training_indices:
            training = SequenceDataset(
                ssm,
                target,
                static_properties,
                dates,
                target_start=0,
                target_stop=train_stop,
                pixel_id=row_index,
            )
            temporal_validation = SequenceDataset(
                ssm,
                target,
                static_properties,
                dates,
                target_start=train_stop,
                target_stop=validation_stop,
                pixel_id=row_index,
            )
            if len(training):
                training_datasets.append(training)
            if len(temporal_validation):
                temporal_validation_datasets.append(temporal_validation)
        else:
            spatial_validation = SequenceDataset(
                ssm,
                target,
                static_properties,
                dates,
                target_start=0,
                target_stop=validation_stop,
                pixel_id=row_index,
            )
            if len(spatial_validation):
                spatial_validation_datasets.append(spatial_validation)

    if not all(
        (
            training_datasets,
            temporal_validation_datasets,
            spatial_validation_datasets,
        )
    ):
        raise RuntimeError(
            f"One or more LSTM data splits are empty for {product}."
        )
    return {
        "training": ConcatDataset(training_datasets),
        "temporal_validation": temporal_validation_datasets,
        "spatial_validation": spatial_validation_datasets,
    }
