"""Leakage-safe sensor-level EF time-constant calibration."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .filter import filtered_rzsm


PRODUCTS = ("ASCAT", "SMAP")
TARGET_VARIABLE = "in-situ_RZSM"
FIXED_T_DAYS = 15.0
DEFAULT_CANDIDATES = np.arange(1.0, 101.0)
SCORE_START = np.datetime64("2016-03-31", "D")
SCORE_END = np.datetime64("2022-04-20", "D")


def _correlation(observed, predicted, minimum_paired):
    paired = np.isfinite(observed) & np.isfinite(predicted)
    count = int(paired.sum())
    if count < int(minimum_paired):
        return np.nan, count
    x, y = observed[paired], predicted[paired]
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return np.nan, count
    return float(np.corrcoef(x, y)[0, 1]), count


def _choose_candidate(candidates, scores):
    finite = np.isfinite(scores)
    if not finite.any():
        return np.nan
    best = np.nanmax(scores)
    tied = candidates[finite & np.isclose(scores, best, rtol=0.0, atol=1e-12)]
    order = np.lexsort((tied, np.abs(tied - FIXED_T_DAYS)))
    return float(tied[order[0]])


def _global_candidate(pixel_optima, candidates):
    valid = np.asarray(pixel_optima, dtype=float)
    valid = valid[np.isfinite(valid)]
    if not len(valid):
        raise RuntimeError("No development pixel produced a valid EF optimum.")
    median = float(np.median(valid))
    distance = np.abs(candidates - median)
    tied = candidates[np.isclose(distance, distance.min(), rtol=0.0, atol=1e-12)]
    order = np.lexsort((tied, np.abs(tied - FIXED_T_DAYS)))
    return float(tied[order[0]])


def calibrate_product(
    product,
    cohort_csv,
    point_directory,
    candidates=DEFAULT_CANDIDATES,
    minimum_paired=100,
):
    """Select one transferable T from per-pixel development optima."""
    product = str(product).upper()
    if product not in PRODUCTS:
        raise ValueError(f"Unsupported product {product!r}.")
    candidates = np.unique(np.asarray(candidates, dtype=float))
    if not len(candidates) or np.any(~np.isfinite(candidates)) or np.any(candidates <= 0):
        raise ValueError("Candidate T values must be finite and positive.")
    cohort = pd.read_csv(cohort_csv).drop_duplicates(["lat_idx", "lon_idx"])
    candidate_rows, optimum_rows = [], []

    for row in cohort.itertuples(index=False):
        lat_idx, lon_idx = int(row.lat_idx), int(row.lon_idx)
        point_file = Path(point_directory) / f"{lat_idx}_{lon_idx}.nc"
        with xr.open_dataset(point_file) as dataset:
            required = ("time", f"{product}_SSM", TARGET_VARIABLE)
            missing = [name for name in required if name not in dataset]
            if missing:
                raise KeyError(f"{point_file} is missing {missing}.")
            dates = np.asarray(dataset["time"].values).astype("datetime64[D]")
            ssm = np.asarray(dataset[f"{product}_SSM"].values).squeeze().astype(float)
            target = np.asarray(dataset[TARGET_VARIABLE].values).squeeze().astype(float)
        if dates.ndim != 1 or ssm.shape != dates.shape or target.shape != dates.shape:
            raise ValueError(f"Unexpected point-array shape in {point_file}.")
        target[(target < 0.0) | (target > 1.0)] = np.nan
        score_period = (dates >= SCORE_START) & (dates <= SCORE_END)
        scores, counts = [], []
        for candidate in candidates:
            prediction = filtered_rzsm(ssm, dates, candidate)
            score, count = _correlation(
                target[score_period], prediction[score_period], minimum_paired
            )
            scores.append(score)
            counts.append(count)
            candidate_rows.append(
                {"product": product, "lat_idx": lat_idx, "lon_idx": lon_idx,
                 "T_days": candidate, "Pearson_R": score, "paired_count": count}
            )
        scores = np.asarray(scores)
        optimum = _choose_candidate(candidates, scores)
        optimum_index = int(np.flatnonzero(candidates == optimum)[0]) if np.isfinite(optimum) else None
        optimum_rows.append(
            {"product": product, "lat_idx": lat_idx, "lon_idx": lon_idx,
             "optimal_T_days": optimum,
             "optimal_Pearson_R": scores[optimum_index] if optimum_index is not None else np.nan,
             "paired_count": counts[optimum_index] if optimum_index is not None else 0}
        )

    candidate_frame = pd.DataFrame(candidate_rows)
    optimum_frame = pd.DataFrame(optimum_rows)
    selected = _global_candidate(optimum_frame["optimal_T_days"], candidates)
    summary = {
        "product": product,
        "selected_global_T_days": selected,
        "selection": "median of development-pixel optimal T, snapped to candidate grid",
        "score_metric": "Pearson correlation",
        "score_start": str(SCORE_START),
        "score_end": str(SCORE_END),
        "minimum_paired": int(minimum_paired),
        "development_pixels": int(len(optimum_frame)),
        "valid_pixel_optima": int(optimum_frame["optimal_T_days"].notna().sum()),
        "independent_test_targets_used": False,
    }
    return candidate_frame, optimum_frame, summary


def save_calibration(output_directory, candidates, optima, summaries):
    """Atomically save calibration tables and the frozen sensor-level values."""
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    files = {
        "candidate_scores": output / "Calibrated_EF_candidate_scores.csv",
        "pixel_optima": output / "Calibrated_EF_pixel_optimal_T.csv",
        "summary": output / "Calibrated_EF_T_summary.json",
    }
    for frame, path in ((candidates, files["candidate_scores"]), (optima, files["pixel_optima"])):
        temporary = path.with_name(f".{path.name}.tmp")
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    payload = {
        "run_complete": True,
        "policy": "development_only_sensor_specific_global_T",
        "fixed_EF_T_days": FIXED_T_DAYS,
        "independent_test_targets_used": False,
        "products": {item["product"]: item for item in summaries},
    }
    temporary = files["summary"].with_name(f".{files['summary'].name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, files["summary"])
    return files


def load_calibrated_t(summary_file, product):
    """Load and validate one frozen sensor-level calibrated T."""
    payload = json.loads(Path(summary_file).read_text())
    if not payload.get("run_complete") or payload.get("independent_test_targets_used") is not False:
        raise RuntimeError(f"Incomplete or unsafe calibration summary: {summary_file}")
    return float(payload["products"][str(product).upper()]["selected_global_T_days"])
