"""Calculate, export, and summarize point-scale EF predictions."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import xarray as xr

from .filter import filtered_rzsm


SUMMARY_COLUMNS = [
    "lat_idx",
    "lon_idx",
    "lat",
    "lon",
    "R",
    "RMSE",
    "KGE",
    "Bias",
    "Flag",
]


def _validate_point_file_dates(
    input_file,
    expected_dates,
):
    """Require one point file to use the exact expected daily coordinate."""
    with xr.open_dataset(input_file, decode_times=True) as dataset:
        if "time" not in dataset.coords:
            raise ValueError(f"{input_file} has no time coordinate")
        input_dates = pd.DatetimeIndex(
            pd.to_datetime(dataset["time"].values)
        ).normalize()

    if not input_dates.equals(expected_dates):
        missing = expected_dates.difference(input_dates)
        extra = input_dates.difference(expected_dates)
        duplicate_count = int(input_dates.duplicated().sum())
        raise ValueError(
            f"{input_file} does not match the required study period: "
            f"count={len(input_dates)}, start={input_dates.min().date()}, "
            f"end={input_dates.max().date()}, missing={len(missing)}, "
            f"extra={len(extra)}, duplicates={duplicate_count}"
        )


def canonical_pixel_files(
    canonical_csv,
    input_directory,
    expected_start=None,
    expected_end=None,
    expected_count=None,
):
    """Return canonical filenames after cohort, file, and date validation."""
    if not os.path.exists(canonical_csv):
        raise FileNotFoundError(
            f"Canonical point cohort was not found: {canonical_csv}"
        )
    if not os.path.isdir(input_directory):
        raise FileNotFoundError(
            f"ISMN input folder was not found: {input_directory}"
        )

    canonical_df = pd.read_csv(canonical_csv)
    required_columns = {"lat_idx", "lon_idx"}
    if not required_columns.issubset(canonical_df.columns):
        raise KeyError(
            f"{canonical_csv} is missing columns {sorted(required_columns)}"
        )
    if canonical_df.duplicated(["lat_idx", "lon_idx"]).any():
        raise RuntimeError(f"Duplicate pixel keys in {canonical_csv}")

    filenames = sorted(
        f"{int(row.lat_idx)}_{int(row.lon_idx)}.nc"
        for row in canonical_df.itertuples()
    )
    if not filenames:
        raise RuntimeError(f"Canonical point cohort is empty: {canonical_csv}")

    missing_inputs = [
        filename
        for filename in filenames
        if not os.path.exists(os.path.join(input_directory, filename))
    ]
    if missing_inputs:
        raise FileNotFoundError(
            f"{len(missing_inputs)} canonical point files are missing from "
            f"{input_directory}; first={missing_inputs[0]}"
        )

    date_arguments = (expected_start, expected_end, expected_count)
    if any(value is not None for value in date_arguments):
        if expected_start is None or expected_end is None:
            raise ValueError(
                "expected_start and expected_end must be supplied together"
            )
        expected_dates = pd.date_range(
            pd.Timestamp(expected_start).normalize(),
            pd.Timestamp(expected_end).normalize(),
            freq="D",
        )
        if expected_count is not None and len(expected_dates) != expected_count:
            raise ValueError(
                f"Expected date count is {len(expected_dates)}, not "
                f"{expected_count}"
            )
        for filename in filenames:
            _validate_point_file_dates(
                os.path.join(input_directory, filename),
                expected_dates,
            )
    return filenames


def fixed_t_station_metrics(observed_rzsm, simulated_rzsm):
    """Calculate legacy EF station metrics after the 0 < RZSM <= 1 mask."""
    observed = np.asarray(observed_rzsm, dtype=float).reshape(-1)
    simulated = np.asarray(simulated_rzsm, dtype=float).reshape(-1)
    observed = np.where(
        (observed > 0.0) & (observed <= 1.0),
        observed,
        np.nan,
    )
    simulated = np.where(
        (simulated > 0.0) & (simulated <= 1.0),
        simulated,
        np.nan,
    )

    valid_simulation = np.isfinite(simulated)
    paired = valid_simulation & np.isfinite(observed)
    metrics = {"R": np.nan, "RMSE": np.nan, "KGE": np.nan, "Bias": np.nan}

    if not valid_simulation.any():
        return simulated, metrics, "Warning: EF RZSM has no valid data"
    if np.count_nonzero(paired) < 2:
        return simulated, metrics, "Warning: Too few paired samples"

    observed_paired = observed[paired]
    simulated_paired = simulated[paired]
    observed_sigma = np.std(observed_paired)
    simulated_sigma = np.std(simulated_paired)
    if observed_sigma == 0 or simulated_sigma == 0:
        return simulated, metrics, "Warning: Zero variance in paired samples"

    correlation = np.corrcoef(observed_paired, simulated_paired)[0, 1]
    rmse = np.sqrt(np.mean((observed_paired - simulated_paired) ** 2))
    observed_mean = np.mean(observed_paired)
    simulated_mean = np.mean(simulated_paired)
    beta = simulated_mean / observed_mean if observed_mean != 0 else 0
    gamma = simulated_sigma / observed_sigma if observed_sigma != 0 else 0
    kge = 1 - np.sqrt(
        (correlation - 1) ** 2 + (beta - 1) ** 2 + (gamma - 1) ** 2
    )
    metrics.update(
        {
            "R": correlation,
            "RMSE": rmse,
            "KGE": kge,
            "Bias": simulated_mean - observed_mean,
        }
    )
    return simulated, metrics, ""


def process_station_file(task):
    """Write one canonical EF prediction file and return its station metrics."""
    filename = task["filename"]
    pixel_name = os.path.splitext(filename)[0]
    try:
        latitude_index, longitude_index = map(int, pixel_name.split("_"))
    except ValueError as exc:
        raise ValueError(f"Unexpected point filename: {filename}") from exc

    latitude = round(float(task["latitude_axis"][latitude_index]), 2)
    longitude = round(float(task["longitude_axis"][longitude_index]), 2)
    input_file = os.path.join(task["input_directory"], filename)
    data_type = task["data_type"]
    surface_variable = f"{data_type}_SSM"
    target_variable = task["target_variable"]

    with xr.open_dataset(input_file) as dataset:
        required_variables = [surface_variable, target_variable]
        missing_variables = [
            variable for variable in required_variables if variable not in dataset
        ]
        if missing_variables:
            raise KeyError(
                f"{input_file} is missing variables: {', '.join(missing_variables)}"
            )
        surface_ssm = np.asarray(dataset[surface_variable].values).squeeze().reshape(-1)
        observed_rzsm = (
            np.asarray(dataset[target_variable].values).squeeze().reshape(-1)
        )
        raw_time = np.asarray(dataset["time"].values).reshape(-1)

    predicted_rzsm = filtered_rzsm(
        surface_ssm,
        raw_time,
        task["filter_time_days"],
        task["spin_up_days"],
    ).reshape(-1)

    lengths = {
        len(raw_time),
        len(surface_ssm),
        len(observed_rzsm),
        len(predicted_rzsm),
    }
    if len(lengths) != 1:
        raise ValueError(
            f"Time and soil-moisture lengths differ in {input_file}: "
            f"time={len(raw_time)}, SSM={len(surface_ssm)}, "
            f"RZSM={len(observed_rzsm)}, EF={len(predicted_rzsm)}"
        )
    study_dates = pd.DatetimeIndex(pd.to_datetime(raw_time)).normalize()

    cleaned_prediction, metrics, flag = fixed_t_station_metrics(
        observed_rzsm,
        predicted_rzsm,
    )
    output_dataset = xr.Dataset(
        data_vars={
            "SSM": ("time", surface_ssm),
            "RZSM": ("time", observed_rzsm),
            "RZSM_prediction": ("time", cleaned_prediction),
        },
        coords={"time": raw_time},
        attrs={
            "source_point_file": input_file,
            "source_prediction_variable": surface_variable,
            "prediction_source": "calculated from source SSM",
            "target_variable": target_variable,
            "filter_T_days": float(task["filter_time_days"]),
            "filter_policy": str(task.get("filter_policy", "fixed_T15")),
            "independent_test_targets_used_for_T_selection": np.int8(0),
            "adjustment_period": "one year",
            "adjustment_days_for_fixed_calendar": float(task["spin_up_days"]),
            "study_period_start": str(study_dates[0].date()),
            "study_period_end": str(study_dates[-1].date()),
            "study_period_days": int(len(study_dates)),
            "output_support": (
                f"valid {data_type} SSM observation dates only"
            ),
        },
    )

    output_file = os.path.join(
        task["output_directory"],
        f"{latitude_index}_{longitude_index}_prediction.nc",
    )
    temporary_file = f"{output_file}.tmp.{os.getpid()}"
    try:
        output_dataset.to_netcdf(temporary_file)
        os.replace(temporary_file, output_file)
    finally:
        output_dataset.close()
        if os.path.exists(temporary_file):
            os.remove(temporary_file)

    return {
        "lat_idx": latitude_index,
        "lon_idx": longitude_index,
        "lat": latitude,
        "lon": longitude,
        "R": round(metrics["R"], 4),
        "RMSE": round(metrics["RMSE"], 4),
        "KGE": round(metrics["KGE"], 4),
        "Bias": round(metrics["Bias"], 4),
        "Flag": flag,
    }


def prediction_run_status(filenames, results, output_directory):
    """Summarize expected, written, warning, and missing canonical pixels."""
    expected_pixels = sorted(
        os.path.splitext(filename)[0] for filename in filenames
    )
    succeeded_pixels = sorted(
        pixel
        for pixel in expected_pixels
        if os.path.exists(os.path.join(output_directory, f"{pixel}_prediction.nc"))
    )
    missing_outputs = sorted(set(expected_pixels) - set(succeeded_pixels))
    warning_pixels = sorted(
        f"{int(result['lat_idx'])}_{int(result['lon_idx'])}"
        for result in results
        if str(result["Flag"]).startswith("Warning:")
    )
    return {
        "expected_pixels": expected_pixels,
        "succeeded_pixels": succeeded_pixels,
        "warning_pixels": warning_pixels,
        "missing_outputs": missing_outputs,
    }
