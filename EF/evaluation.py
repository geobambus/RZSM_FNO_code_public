"""Read EF prediction files and calculate paired RZSM evaluation metrics."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import xarray as xr


def period_mask(dates, start_date=None, end_date=None):
    """Return a boolean mask for an inclusive evaluation period."""
    date_values = np.asarray(dates, dtype="datetime64[ns]")
    mask = np.ones(len(date_values), dtype=bool)
    if start_date is not None:
        mask &= date_values >= np.datetime64(pd.Timestamp(start_date), "ns")
    if end_date is not None:
        mask &= date_values <= np.datetime64(pd.Timestamp(end_date), "ns")
    return mask


def comparison_metrics(observed, predicted):
    """Return R, RMSE, bias, and unbiased RMSE for finite paired values."""
    observed = np.asarray(observed, dtype=float).reshape(-1)
    predicted = np.asarray(predicted, dtype=float).reshape(-1)
    paired = np.isfinite(observed) & np.isfinite(predicted)
    observed_paired = observed[paired]
    predicted_paired = predicted[paired]
    count = len(observed_paired)

    if count == 0:
        return {
            "R": np.nan,
            "RMSE": np.nan,
            "Bias": np.nan,
            "ubRMSE": np.nan,
            "N": 0,
        }

    rmse = np.sqrt(np.mean((observed_paired - predicted_paired) ** 2))
    bias = np.mean(predicted_paired - observed_paired)
    ubrmse = np.sqrt(np.maximum(0.0, rmse**2 - bias**2))
    correlation = np.nan
    if (
        count >= 2
        and np.std(observed_paired) != 0
        and np.std(predicted_paired) != 0
    ):
        correlation = np.corrcoef(observed_paired, predicted_paired)[0, 1]
    return {
        "R": correlation,
        "RMSE": rmse,
        "Bias": bias,
        "ubRMSE": ubrmse,
        "N": count,
    }


def read_prediction_file(path):
    """Read time, SSM, observed RZSM, and EF RZSM from one point file."""
    with xr.open_dataset(path) as dataset:
        required = ["time", "SSM", "RZSM", "RZSM_prediction"]
        missing = [name for name in required if name not in dataset]
        if missing:
            raise KeyError(f"{path} is missing variables: {', '.join(missing)}")
        values = {
            "time": pd.to_datetime(dataset["time"].values).to_numpy(),
            "SSM": np.asarray(dataset["SSM"].values, dtype=float).reshape(-1),
            "RZSM": np.asarray(dataset["RZSM"].values, dtype=float).reshape(-1),
            "RZSM_prediction": np.asarray(
                dataset["RZSM_prediction"].values,
                dtype=float,
            ).reshape(-1),
        }

    minimum_length = min(len(array) for array in values.values())
    return {name: array[:minimum_length] for name, array in values.items()}


def evaluate_pixel_task(task):
    """Return finite paired test-period values for one scatterplot task."""
    latitude_index = int(task["lat_idx"])
    longitude_index = int(task["lon_idx"])
    prediction_file = os.path.join(
        task["prediction_directory"],
        f"{latitude_index}_{longitude_index}_prediction.nc",
    )
    if not os.path.exists(prediction_file):
        return None

    series = read_prediction_file(prediction_file)
    selected = period_mask(
        series["time"],
        start_date=task["start_date"],
        end_date=task["end_date"],
    )
    paired = (
        selected
        & np.isfinite(series["RZSM"])
        & np.isfinite(series["RZSM_prediction"])
    )
    if np.count_nonzero(paired) < 2:
        return None
    return {
        "lat_idx": latitude_index,
        "lon_idx": longitude_index,
        "observed": series["RZSM"][paired],
        "predicted": series["RZSM_prediction"][paired],
    }


def select_timeseries_file(path, start_date=None, end_date=None):
    """Read one EF point file and select an inclusive plotting period."""
    series = read_prediction_file(path)
    selected = period_mask(series["time"], start_date, end_date)
    return {
        name: values[selected]
        for name, values in series.items()
    }


def evaluate_timeseries_file(path, start_date=None, end_date=None):
    """Select a period and calculate metrics for one EF prediction file."""
    selected_series = select_timeseries_file(path, start_date, end_date)
    metrics = comparison_metrics(
        selected_series["RZSM"],
        selected_series["RZSM_prediction"],
    )
    return selected_series, metrics
