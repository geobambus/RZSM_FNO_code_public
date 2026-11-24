"""Shared diagnostics for pixel-scale multi-seed RZSM estimations."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd


TRIAL_VARIABLE_PATTERN = re.compile(r"^RZSM_trial_(\d+)$")
SPREAD_DIAGNOSTIC_COLUMNS = (
    "RZSM_std_mean",
    "RZSM_std_median",
    "RZSM_std_q95",
    "RZSM_std_max",
    "RZSM_std_abs_error_spearman",
    "N_std",
)


def _as_float_array(values):
    return np.ma.filled(values, np.nan).astype(float).ravel()


def trial_variable_names(dataset):
    """Return numerically ordered, contiguous RZSM trial variable names."""
    indexed = []
    for name in dataset.variables:
        match = TRIAL_VARIABLE_PATTERN.match(name)
        if match:
            indexed.append((int(match.group(1)), name))
    indexed.sort()
    if not indexed:
        raise KeyError("Estimation file has no RZSM_trial_* variables.")

    indices = [index for index, _ in indexed]
    expected_indices = list(range(1, len(indexed) + 1))
    if indices != expected_indices:
        raise ValueError(
            f"RZSM trial variables are not contiguous: {indices}; "
            f"expected {expected_indices}."
        )

    declared_count = int(getattr(dataset, "trial_count", len(indexed)))
    if declared_count != len(indexed):
        raise ValueError(
            f"Estimation file declares {declared_count} trials but contains "
            f"{len(indexed)} RZSM_trial_* variables."
        )
    return [name for _, name in indexed]


def read_seed_ensemble(dataset, *, validate=True):
    """Read trial, ensemble-mean, and seed-spread arrays from a NetCDF file."""
    required = ("RZSM_avg", "RZSM_std")
    missing = [name for name in required if name not in dataset.variables]
    if missing:
        raise KeyError(f"Estimation file is missing ensemble variables: {missing}")

    names = trial_variable_names(dataset)
    trials = [_as_float_array(dataset.variables[name][:]) for name in names]
    ensemble_mean = _as_float_array(dataset.variables["RZSM_avg"][:])
    seed_spread = _as_float_array(dataset.variables["RZSM_std"][:])
    seeds = [
        int(getattr(dataset.variables[name], "seed", index))
        for index, name in enumerate(names, start=1)
    ]

    lengths = {len(ensemble_mean), len(seed_spread), *(len(values) for values in trials)}
    if len(lengths) != 1:
        raise ValueError(f"Ensemble variables do not share one time dimension: {sorted(lengths)}")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"Estimation file has duplicate seed metadata: {seeds}")

    output = {
        "trial_names": names,
        "seeds": seeds,
        "trials": trials,
        "ensemble_mean": ensemble_mean,
        "seed_spread": seed_spread,
    }
    if validate:
        validate_seed_ensemble(output)
    return output


def validate_seed_ensemble(ensemble, *, atol=5e-6, rtol=5e-5):
    """Require shared missing dates and mean/sample spread matching every seed."""
    trial_matrix = np.vstack(ensemble["trials"])
    ensemble_mean = np.asarray(ensemble["ensemble_mean"], dtype=float)
    seed_spread = np.asarray(ensemble["seed_spread"], dtype=float)
    if any(np.isinf(values).any() for values in (trial_matrix, ensemble_mean, seed_spread)):
        raise ValueError("Ensemble variables contain infinite values; use NaN for missing dates.")
    if np.any(seed_spread < 0.0):
        raise ValueError("RZSM_std contains negative values.")

    finite_trials = np.isfinite(trial_matrix)
    complete = finite_trials.all(axis=0)
    if not np.array_equal(finite_trials.any(axis=0), complete):
        raise ValueError("RZSM_trial_* do not share the same finite timestamp support.")
    for name, values in (("RZSM_avg", ensemble_mean), ("RZSM_std", seed_spread)):
        if not np.array_equal(np.isfinite(values), complete):
            raise ValueError(f"{name} does not share the finite timestamp support of RZSM_trial_*.")
    if not np.any(complete):
        raise ValueError("Estimation file has no timestamps with a complete seed ensemble.")

    expected_mean = np.mean(trial_matrix[:, complete], axis=0)
    expected_std = np.std(trial_matrix[:, complete], axis=0, ddof=1)
    if not np.allclose(ensemble_mean[complete], expected_mean, atol=atol, rtol=rtol):
        raise ValueError("Stored RZSM_avg does not match the mean of RZSM_trial_*.")
    if not np.allclose(seed_spread[complete], expected_std, atol=atol, rtol=rtol):
        raise ValueError(
            "Stored RZSM_std does not match the sample standard deviation "
            "(ddof=1) of RZSM_trial_*."
        )


def calculate_estimation_metrics(observed, estimated):
    """Calculate the paper's pixel-scale deterministic metrics."""
    observed = np.asarray(observed, dtype=float)
    estimated = np.asarray(estimated, dtype=float)
    valid = np.isfinite(observed) & np.isfinite(estimated)
    observed = observed[valid]
    estimated = estimated[valid]
    if len(observed) == 0:
        return {
            "R": np.nan,
            "RMSE": np.nan,
            "bias": np.nan,
            "ubRMSE": np.nan,
            "N": 0,
        }

    if len(observed) > 1 and np.std(observed) > 0.0 and np.std(estimated) > 0.0:
        correlation = float(np.corrcoef(observed, estimated)[0, 1])
    else:
        correlation = np.nan
    residual = estimated - observed
    rmse = float(np.sqrt(np.mean(residual**2)))
    bias = float(np.mean(residual))
    return {
        "R": correlation,
        "RMSE": rmse,
        "bias": bias,
        "ubRMSE": float(np.sqrt(max(0.0, rmse**2 - bias**2))),
        "N": int(len(observed)),
    }


def spearman_rank_correlation(first, second):
    """Return Spearman's rho without interpreting it as a significance test."""
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    valid = np.isfinite(first) & np.isfinite(second)
    first = first[valid]
    second = second[valid]
    if len(first) < 3 or np.unique(first).size < 2 or np.unique(second).size < 2:
        return np.nan
    first_rank = pd.Series(first).rank(method="average").to_numpy(dtype=float)
    second_rank = pd.Series(second).rank(method="average").to_numpy(dtype=float)
    return float(np.corrcoef(first_rank, second_rank)[0, 1])


def summarize_seed_spread(observed, ensemble_mean, seed_spread):
    """Summarize initialization-seed spread and its descriptive error relation."""
    observed = np.asarray(observed, dtype=float)
    ensemble_mean = np.asarray(ensemble_mean, dtype=float)
    seed_spread = np.asarray(seed_spread, dtype=float)
    valid = np.isfinite(observed) & np.isfinite(ensemble_mean) & np.isfinite(seed_spread)
    valid &= seed_spread >= 0.0
    if not np.any(valid):
        return {name: np.nan for name in SPREAD_DIAGNOSTIC_COLUMNS[:-1]} | {"N_std": 0}

    spread = seed_spread[valid]
    absolute_error = np.abs(ensemble_mean[valid] - observed[valid])
    return {
        "RZSM_std_mean": float(np.mean(spread)),
        "RZSM_std_median": float(np.median(spread)),
        "RZSM_std_q95": float(np.quantile(spread, 0.95)),
        "RZSM_std_max": float(np.max(spread)),
        "RZSM_std_abs_error_spearman": spearman_rank_correlation(
            spread, absolute_error
        ),
        "N_std": int(len(spread)),
    }
