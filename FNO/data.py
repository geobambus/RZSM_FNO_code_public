"""Point-series preparation and PyTorch datasets for FNO training."""

import os

import netCDF4 as nc
import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, Dataset

from LSTM.data import apply_scaler, compute_scaler, spatial_training_indices

from .settings import (
    PRODUCT_SETTINGS,
    TARGET_VARIABLE,
    WINDOW_SIZE,
    build_observation_sequences,
    read_point_series,
    split_boundaries,
    validate_product,
    validate_static_columns,
)


class SequenceDataset(Dataset):
    """Observation-event sequences for one point and target interval."""

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


def build_training_datasets(input_csv, ismn_root, product, scaler):
    """Build the active temporal and spatial point splits for FNO."""
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
        raise RuntimeError(f"One or more FNO data splits are empty for {product}.")
    return {
        "training": ConcatDataset(training_datasets),
        "temporal_validation": temporal_validation_datasets,
        "spatial_validation": spatial_validation_datasets,
    }


__all__ = [
    "SequenceDataset",
    "apply_scaler",
    "build_training_datasets",
    "compute_scaler",
    "spatial_training_indices",
]
