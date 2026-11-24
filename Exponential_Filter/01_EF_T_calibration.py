#!/usr/bin/env python3
"""Step 1: calibrate one global EF time constant T per satellite (SMAP, ASCAT).

Following Pasik et al. (2023, GMD 16:4957, Sect. 3.1.2), each SCAN training
pixel's optimal T maximizes the Pearson R between filtered SSM and in situ
RZSM over T = 1-100 d. Pixels whose best-T R is below 0.5 or not significant
(p >= 0.05) are left out, and the global T is the median optimum of the
remaining pixels. Only development dates before the test period are scored;
test-period targets are never read.

SMAP and ASCAT share the in situ target, the pixels and the selection rule;
as in the gridded EF, 0 is a valid ASCAT value. The selected values are the
T used by every EF product (``core.rzsm_pipeline_config.load_ef_t``).
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
import json
import math
import os

PUBLICATION_CODE = Path(__file__).resolve().parents[1]
if str(PUBLICATION_CODE) not in sys.path:
    sys.path.insert(0, str(PUBLICATION_CODE))

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

from core.project_paths import PATHS
from core.rzsm_pipeline_config import (
    EF_CALIBRATION_CANDIDATE_DAYS,
    EF_CALIBRATION_COHORT,
    EF_CALIBRATION_MAX_P,
    EF_CALIBRATION_MIN_PAIRED,
    EF_CALIBRATION_MIN_R,
    EF_CALIBRATION_SCORE_END_DATE,
    EF_CALIBRATION_SCORE_START_DATE,
    EF_CALIBRATION_SCREENING,
    EF_CALIBRATION_TIE_T_DAYS,
    EF_POLICY,
    EF_SPIN_UP_DAYS,
    PIPELINE_VERSION,
    load_ef_t,
)


PRODUCTS = ("SMAP", "ASCAT")
TARGET_VARIABLE = "in-situ_RZSM"
SSM_MINIMUM_INCLUSIVE = {"SMAP": False, "ASCAT": True}


def calibration_settings():
    ismn = PATHS.results / "International_Soil_Moisture_Network"
    return SimpleNamespace(
        data=PRODUCTS,
        pixel_root=ismn / "ISMN_Train",
        cohort_csv=ismn / "Station_TCA_Screening" / "Common_Train_pixel_list.csv",
        output_root=PATHS.results / "Exponential_Filter" / "T_calibration",
        candidate_days=np.asarray(EF_CALIBRATION_CANDIDATE_DAYS, dtype=float),
        min_paired=EF_CALIBRATION_MIN_PAIRED,
    )


def exponential_filter(ssm, time_days, T):
    """Update filter state at valid SSM observations, retaining gaps as NaN."""
    values = np.asarray(ssm, dtype=float)
    times = np.asarray(time_days, dtype=float)
    if values.ndim != 1 or values.shape != times.shape:
        raise ValueError('SSM and time must be aligned one-dimensional arrays.')
    if not np.isfinite(T) or T <= 0:
        raise ValueError('T must be finite and positive.')
    if not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
        raise ValueError('Observation times must be finite and strictly increasing.')

    output = np.full(values.shape, np.nan, dtype=float)
    valid = np.flatnonzero(np.isfinite(values))
    if valid.size == 0:
        return output
    first = int(valid[0])
    state = float(values[first])
    previous_gain = 1.0
    previous_time = float(times[first])
    output[first] = state
    for index in valid[1:]:
        elapsed = float(times[index] - previous_time)
        decay = np.exp(-elapsed / float(T))
        gain = previous_gain / (previous_gain + decay)
        state += gain * (float(values[index]) - state)
        output[index] = state
        previous_gain = gain
        previous_time = float(times[index])
    return output


def pearson_r(observed: np.ndarray, simulated: np.ndarray, minimum: int) -> tuple[float, int]:
    valid = np.isfinite(observed) & np.isfinite(simulated)
    count = int(np.count_nonzero(valid))
    if count < minimum:
        return np.nan, count
    x = np.asarray(observed[valid], dtype=float)
    y = np.asarray(simulated[valid], dtype=float)
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return np.nan, count
    return float(np.corrcoef(x, y)[0, 1]), count


def correlation_p_value(r: float, count: int) -> float:
    """Two-sided p of a Pearson R through Fisher's z transform."""
    if not np.isfinite(r) or count < 4:
        return np.nan
    if abs(r) >= 1.0:
        return 0.0
    z = math.atanh(r) * math.sqrt(count - 3)
    return math.erfc(abs(z) / math.sqrt(2.0))


def passes_screening(r: float, p_value: float) -> bool:
    return bool(np.isfinite(r) and r >= EF_CALIBRATION_MIN_R
                and np.isfinite(p_value) and p_value < EF_CALIBRATION_MAX_P)


def choose_candidate(candidates: np.ndarray, scores: np.ndarray) -> float:
    """Select max R, breaking numerical ties toward T=15 and then lower T."""
    finite = np.isfinite(scores)
    if not np.any(finite):
        return np.nan
    best = np.nanmax(scores)
    tied = candidates[finite & np.isclose(scores, best, rtol=0.0, atol=1e-12)]
    order = np.lexsort((tied, np.abs(tied - EF_CALIBRATION_TIE_T_DAYS)))
    return float(tied[order[0]])


def snap_median_to_candidate(pixel_optima: np.ndarray, candidates: np.ndarray) -> float:
    finite = pixel_optima[np.isfinite(pixel_optima)]
    if finite.size == 0:
        raise RuntimeError("No pixel produced a valid EF T optimum")
    target = float(np.median(finite))
    distance = np.abs(candidates - target)
    tied = candidates[np.isclose(distance, np.min(distance), rtol=0.0, atol=1e-12)]
    order = np.lexsort((tied, np.abs(tied - EF_CALIBRATION_TIE_T_DAYS)))
    return float(tied[order[0]])


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_json(payload: dict, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as target:
        json.dump(payload, target, indent=2, sort_keys=True)
        target.write("\n")
    os.replace(temporary, path)


def load_pixel(
    path: Path,
    product: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with xr.open_dataset(path) as dataset:
        required = ("time", f"{product}_SSM", TARGET_VARIABLE)
        missing = [name for name in required if name not in dataset]
        if missing:
            raise KeyError(f"{path} is missing {missing}")
        if str(dataset.attrs.get("split", "")) != "Train":
            raise ValueError(f"EF calibration pixel is not labelled Train: {path}")
        networks = {
            value.strip()
            for value in str(dataset.attrs.get("networks", "")).split(";")
            if value.strip()
        }
        if networks != {"SCAN"}:
            raise ValueError(
                f"EF calibration requires SCAN-only pixel files; "
                f"found networks={sorted(networks)} in {path}"
            )
        dates = np.asarray(dataset["time"].values).astype("datetime64[D]")
        ssm = np.asarray(dataset[f"{product}_SSM"].values).squeeze().astype(float)
        target = np.asarray(dataset[TARGET_VARIABLE].values).squeeze().astype(float)
    if dates.ndim != 1 or ssm.shape != dates.shape or target.shape != dates.shape:
        raise ValueError(f"Unexpected pixel-array shape in {path}")
    below = ssm < 0.0 if SSM_MINIMUM_INCLUSIVE[product] else ssm <= 0.0
    ssm[below | (ssm > 1.0)] = np.nan
    target[(target < 0.0) | (target > 1.0)] = np.nan
    if dates.size > 1 and np.any(np.diff(dates) <= np.timedelta64(0, "D")):
        raise ValueError(f"Time coordinate is not strictly increasing in {path}")
    time_days = ((dates - dates[0]) / np.timedelta64(1, "D")).astype(float)
    return dates, time_days, ssm, target


def calibrate_product(product: str, args) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    cohort_path = args.cohort_csv
    if not cohort_path.is_file():
        raise FileNotFoundError(f"Missing SCAN training cohort: {cohort_path}")
    cohort = pd.read_csv(cohort_path)
    required = {"lat_idx", "lon_idx"}
    if not required.issubset(cohort.columns):
        raise KeyError(f"{cohort_path} is missing {sorted(required)}")
    cohort = cohort.drop_duplicates(["lat_idx", "lon_idx"])

    candidate_rows: list[dict] = []
    optimum_rows: list[dict] = []
    for row in tqdm(
        cohort.itertuples(),
        total=len(cohort),
        desc=f"Calibrating EF T ({product})",
        leave=False,
    ):
        lat_idx = int(row.lat_idx)
        lon_idx = int(row.lon_idx)
        pixel_path = args.pixel_root / f"{lat_idx}_{lon_idx}.nc"
        if not pixel_path.is_file():
            raise FileNotFoundError(f"Missing canonical SCAN pixel: {pixel_path}")
        dates, time_days, ssm, target = load_pixel(pixel_path, product)
        score_period = (
            (dates >= EF_CALIBRATION_SCORE_START_DATE)
            & (dates <= EF_CALIBRATION_SCORE_END_DATE)
        )
        scores = np.full(len(args.candidate_days), np.nan, dtype=float)
        paired_counts = np.zeros(len(args.candidate_days), dtype=int)
        for candidate_index, candidate in enumerate(args.candidate_days):
            filtered = exponential_filter(ssm, time_days, float(candidate))
            filtered[time_days < EF_SPIN_UP_DAYS] = np.nan
            score, paired = pearson_r(
                target[score_period],
                filtered[score_period],
                args.min_paired,
            )
            scores[candidate_index] = score
            paired_counts[candidate_index] = paired
            candidate_rows.append(
                {
                    "product": product,
                    "lat_idx": lat_idx,
                    "lon_idx": lon_idx,
                    "T_days": float(candidate),
                    "Pearson_R": score,
                    "paired_count": paired,
                }
            )

        optimum = choose_candidate(args.candidate_days, scores)
        optimum_index = (
            int(np.flatnonzero(args.candidate_days == optimum)[0])
            if np.isfinite(optimum)
            else None
        )
        optimal_r = scores[optimum_index] if optimum_index is not None else np.nan
        optimal_count = paired_counts[optimum_index] if optimum_index is not None else 0
        p_value = correlation_p_value(optimal_r, optimal_count)
        optimum_rows.append(
            {
                "product": product,
                "lat_idx": lat_idx,
                "lon_idx": lon_idx,
                "optimal_T_days": optimum,
                "optimal_Pearson_R": optimal_r,
                "paired_count": optimal_count,
                "p_value": p_value,
                "used_for_selection": passes_screening(optimal_r, p_value),
            }
        )

    candidates_frame = pd.DataFrame(candidate_rows)
    optimum_frame = pd.DataFrame(optimum_rows)
    used = optimum_frame["used_for_selection"].to_numpy(dtype=bool)
    if not used.any():
        raise RuntimeError(f"No {product} pixel passes the screening: {EF_CALIBRATION_SCREENING}")
    global_t = snap_median_to_candidate(
        optimum_frame.loc[used, "optimal_T_days"].to_numpy(dtype=float),
        args.candidate_days,
    )
    summary = {
        "product": product,
        "selected_global_T_days": global_t,
        "selection_statistic": (
            "median of per-pixel optimal T over the screened pixels, snapped to candidate grid"
        ),
        "selection_screening": EF_CALIBRATION_SCREENING,
        "candidate_days": [float(value) for value in args.candidate_days],
        "pixel_count_requested": int(len(cohort)),
        "pixel_count_with_valid_optimum": int(
            optimum_frame["optimal_T_days"].notna().sum()
        ),
        "pixel_count_used_for_selection": int(used.sum()),
        "minimum_paired_count": int(args.min_paired),
        "score_metric": "unscaled Pearson R",
        "tie_break": "closest to literature T=15, then smaller T",
        "score_start_date": str(EF_CALIBRATION_SCORE_START_DATE),
        "score_end_date": str(EF_CALIBRATION_SCORE_END_DATE),
        "cohort": EF_CALIBRATION_COHORT,
        "ssm_valid_range": "[0, 1]" if SSM_MINIMUM_INCLUSIVE[product] else "(0, 1]",
    }
    return candidates_frame, optimum_frame, summary


def main():
    args = calibration_settings()
    args.output_root.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_root / "EF_T_calibration_summary.json"

    all_candidates = []
    all_optima = []
    product_summaries = {}
    for product in args.data:
        candidates, optima, summary = calibrate_product(product, args)
        all_candidates.append(candidates)
        all_optima.append(optima)
        product_summaries[product] = summary

    payload = {
        "run_complete": True,
        "pipeline_version": PIPELINE_VERSION,
        "policy": EF_POLICY,
        "products": product_summaries,
        "independent_test_target_values_used": False,
    }
    atomic_csv(
        pd.concat(all_candidates, ignore_index=True),
        args.output_root / "EF_T_calibration_candidate_scores.csv",
    )
    atomic_csv(
        pd.concat(all_optima, ignore_index=True),
        args.output_root / "EF_T_calibration_pixel_optima.csv",
    )
    atomic_json(payload, summary_path)
    for product, summary in product_summaries.items():
        print(f"{product}: selected T = {summary['selected_global_T_days']:g} d "
              f"(load_ef_t uses {load_ef_t(None, product):g} d)")
    return summary_path


if __name__ == "__main__":
    main()
    print("01_EF_T_calibration.py is finished.")
