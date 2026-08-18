"""Create the compact static-feature CSV inputs used by LSTM workflows."""

import os

import netCDF4 as nc
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .settings import (
    AREAS,
    PRODUCT_SETTINGS,
    STATIC_FEATURES,
    TARGET_VARIABLE,
    validate_product,
)


def _finite_scalar(variable, latitude_index, longitude_index):
    value = variable[latitude_index, longitude_index]
    if np.ma.is_masked(value):
        return np.nan
    value = float(value)
    return value if np.isfinite(value) else np.nan


def generate_input_csv(
    product,
    area,
    model_input_file,
    ismn_root,
    output_file,
    min_valid_observations=100,
):
    """Write one area/product pixel inventory with seven static predictors."""
    product = validate_product(product)
    if area not in AREAS:
        raise ValueError(f"Unsupported area {area!r}; expected {AREAS}.")

    pixel_list_file = os.path.join(
        ismn_root,
        "Station_TCA_Screening",
        f"Common_{area}_pixel_list.csv",
    )
    if not os.path.exists(pixel_list_file):
        if area == "Excluded":
            return None
        raise FileNotFoundError(f"Missing common pixel list: {pixel_list_file}")
    pixel_list = pd.read_csv(pixel_list_file).drop_duplicates(
        ["lat_idx", "lon_idx"]
    )
    if pixel_list.empty:
        if area == "Excluded":
            return None
        raise RuntimeError(f"Common pixel list is empty: {pixel_list_file}")

    settings = PRODUCT_SETTINGS[product]
    rows = []
    with nc.Dataset(model_input_file) as static_dataset:
        required = ("lat", "lon", *STATIC_FEATURES)
        missing = [name for name in required if name not in static_dataset.variables]
        if missing:
            raise KeyError(f"Static model input is missing variables: {missing}")

        latitudes = static_dataset.variables["lat"][:]
        longitudes = static_dataset.variables["lon"][:]
        for pixel in tqdm(
            pixel_list.itertuples(index=False),
            total=len(pixel_list),
            desc=f"Preparing {area}-{product}",
            leave=False,
        ):
            latitude_index = int(pixel.lat_idx)
            longitude_index = int(pixel.lon_idx)
            static_values = {
                name: _finite_scalar(
                    static_dataset.variables[name],
                    latitude_index,
                    longitude_index,
                )
                for name in STATIC_FEATURES
            }
            if not all(np.isfinite(list(static_values.values()))):
                continue

            point_file = os.path.join(
                ismn_root,
                f"ISMN_{area}",
                settings["folder"],
                f"{latitude_index}_{longitude_index}.nc",
            )
            if not os.path.exists(point_file):
                continue
            with nc.Dataset(point_file) as point_dataset:
                required_point = (settings["ssm_variable"], TARGET_VARIABLE)
                missing_point = [
                    name
                    for name in required_point
                    if name not in point_dataset.variables
                ]
                if missing_point:
                    raise KeyError(f"{point_file} is missing {missing_point}.")
                ssm = np.ma.filled(
                    point_dataset.variables[settings["ssm_variable"]][:], np.nan
                ).astype(float)
                rzsm = np.ma.filled(
                    point_dataset.variables[TARGET_VARIABLE][:], np.nan
                ).astype(float)
            valid_ssm = int(np.count_nonzero(np.isfinite(ssm)))
            valid_rzsm = int(
                np.count_nonzero(np.isfinite(ssm) & np.isfinite(rzsm))
            )
            if min(valid_ssm, valid_rzsm) < int(min_valid_observations):
                continue

            rows.append(
                {
                    "lat_idx": latitude_index,
                    "lon_idx": longitude_index,
                    "lat": float(latitudes[latitude_index]),
                    "lon": float(longitudes[longitude_index]),
                    "valid_SSM": valid_ssm,
                    "valid_RZSM": valid_rzsm,
                    **static_values,
                }
            )

    output = pd.DataFrame(rows)
    if output.empty:
        if area == "Excluded":
            return None
        raise RuntimeError(f"No usable pixels remain for {area}-{product}.")
    output = output[
        [
            "lat_idx",
            "lon_idx",
            "lat",
            "lon",
            "valid_SSM",
            "valid_RZSM",
            *STATIC_FEATURES,
        ]
    ]
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    temporary_file = f"{output_file}.tmp.{os.getpid()}"
    output.to_csv(temporary_file, index=False)
    os.replace(temporary_file, output_file)
    return output
