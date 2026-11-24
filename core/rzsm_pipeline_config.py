"""Dates, splits, observation windows and EF settings shared by all workflows."""

from __future__ import annotations

import json
import os

import netCDF4 as nc
import numpy as np

from core.project_paths import project_results_root

PIPELINE_VERSION = "RZSM_pixel_v10_shared_60d_calendar_context"
MODEL_SEEDS = (42, 111, 222, 333, 444, 555, 666, 777, 888, 999)
MODEL_SEED = MODEL_SEEDS[0]


MODEL_PRODUCTS = ("ASCAT", "SMAP")


EF_INPUT_START_DATE = np.datetime64("2015-04-01", "D")
EF_OUTPUT_START_DATE = np.datetime64("2016-03-31", "D")
NEURAL_OUTPUT_START_DATE = EF_INPUT_START_DATE
COMMON_START_DATE = EF_OUTPUT_START_DATE
COMMON_END_DATE = np.datetime64("2023-12-31", "D")

ANOMALY_WINDOW_DAYS = 31
ANOMALY_SWEEP_WINDOWS = (31, 61, 101, 141)
ESTIMATION_ANOMALY_CONTEXT_DAYS = max(ANOMALY_SWEEP_WINDOWS) // 2

TRAIN_RATIO = 0.70
VAL_RATIO = 0.10
TEST_START_RATIO = round(TRAIN_RATIO + VAL_RATIO, 2)

OBSERVATION_EVENT_SEQUENCE_LENGTH = 32
OBSERVATION_EVENT_GAP_SCALE_DAYS = 32.0
NEURAL_RECIPE_EVENT = "event32"


VALIDATION_START_DATE = np.datetime64("2021-06-15", "D")


TEST_PERIOD_POLICY = "final_20_percent_of_post_spinup_common_period"

TEST_PIXEL_MIN_MATCHED_DAYS = 100

ASCAT_PRIMARY_FILENAME = "ASCAT_2015_2023_conf0_R_eqd_010.nc"
NLDAS_TARGET_FILENAME = (
    "NLDAS_NOAH0125_H_daily_20150401_20231231_"
    "R_eqd_010_CONUS_target_7_28cm.nc"
)
NLDAS_TARGET_VARIABLE = "RZSM_7_28cm"

EF_POLICY = "development_calibrated_satellite_specific_T"
EF_CALIBRATION_TIE_T_DAYS = 15.0
EF_CALIBRATION_CANDIDATE_DAYS = tuple(float(value) for value in range(1, 101))
EF_CALIBRATION_SCORE_START_DATE = EF_OUTPUT_START_DATE

EF_CALIBRATION_MIN_PAIRED = 100
EF_CALIBRATION_COHORT = "SCAN_train_pixels_only"
EF_CALIBRATION_MIN_R = 0.5
EF_CALIBRATION_MAX_P = 0.05
EF_CALIBRATION_SCREENING = (
    f"best-T Pearson r >= {EF_CALIBRATION_MIN_R:g} and two-sided "
    f"p < {EF_CALIBRATION_MAX_P:g} (Pasik et al., 2023)"
)
EF_SPIN_UP_DAYS = float(
    (EF_OUTPUT_START_DATE - EF_INPUT_START_DATE) / np.timedelta64(1, "D")
)

DEFAULT_FNO_MODEL = "FNO"
FNO_MODEL_CHOICES = (DEFAULT_FNO_MODEL,)
DEFAULT_LSTM_MODEL = "LSTM"
LSTM_MODEL_CHOICES = (
    DEFAULT_LSTM_MODEL,
)
NEURAL_MODEL_CHOICES = FNO_MODEL_CHOICES + LSTM_MODEL_CHOICES


def ratio_period_start(start_date, end_date, start_ratio):
    """Return the sample-index boundary for a ratio of an inclusive daily period."""
    if not 0.0 <= float(start_ratio) <= 1.0:
        raise ValueError(f"start_ratio must be between 0 and 1; received {start_ratio}.")
    dates = np.arange(
        np.datetime64(start_date, "D"),
        np.datetime64(end_date, "D") + np.timedelta64(1, "D"),
        dtype="datetime64[D]",
    )
    if len(dates) == 0:
        raise ValueError(f"Empty daily period: {start_date} to {end_date}.")
    boundary = min(int(len(dates) * float(start_ratio)), len(dates) - 1)
    return dates[boundary]


TEST_START_DATE = ratio_period_start(
    COMMON_START_DATE,
    COMMON_END_DATE,
    TEST_START_RATIO,
)

MODEL_HOLDOUT_START_DATE = TEST_START_DATE
ESTIMATION_START_DATE = TEST_START_DATE - np.timedelta64(
    ESTIMATION_ANOMALY_CONTEXT_DAYS, "D"
)
ESTIMATION_YEARS = list(range(
    int(str(ESTIMATION_START_DATE)[:4]), int(str(COMMON_END_DATE)[:4]) + 1
))

EF_CALIBRATION_SCORE_END_DATE = TEST_START_DATE - np.timedelta64(1, "D")


def _numeric_time_days(time_values, length):
    """Convert a monotonic time coordinate to elapsed floating-point days."""
    if time_values is None:
        values = np.arange(length, dtype=np.float64)
    else:
        values = np.asarray(time_values)
        if len(values) != length:
            raise ValueError(
                f"Time/value length mismatch: {len(values)} vs {length}."
            )
        if np.issubdtype(values.dtype, np.datetime64):
            day_values = values.astype("datetime64[D]")
            values = (
                (day_values - day_values[0]) / np.timedelta64(1, "D")
            ).astype(np.float64)
        else:
            values = values.astype(np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("Observation time coordinate contains non-finite values.")
    if len(values) > 1 and np.any(np.diff(values) <= 0):
        raise ValueError("Observation time coordinate must be strictly increasing.")
    return values


def prepare_observation_time_days(time_values, length):
    """Convert a time axis shared by many pixels to elapsed days once."""
    return _numeric_time_days(time_values, length)


def build_observation_event_sequences(
    ssm,
    target=None,
    *,
    time_values=None,
    prepared_time_days=None,
    target_start=0,
    target_stop=None,
    window_size=OBSERVATION_EVENT_SEQUENCE_LENGTH,
    gap_scale_days=OBSERVATION_EVENT_GAP_SCALE_DAYS,
):
    """Build past-only windows from the previous valid satellite retrievals.

    Each returned sequence ends on its estimation timestamp and contains
    ``[SSM, delta_t_days / gap_scale_days]`` for the latest ``window_size``
    valid observations.  The first delta in every window is zero because the
    event immediately before the window is deliberately outside the model's
    causal context.  Missing calendar dates are skipped rather than filled.
    """
    ssm = np.asarray(ssm, dtype=np.float32).reshape(-1)
    n_time = len(ssm)
    window_size = int(window_size)
    gap_scale_days = float(gap_scale_days)
    if window_size < 1:
        raise ValueError(f"window_size must be positive; received {window_size}.")
    if not np.isfinite(gap_scale_days) or gap_scale_days <= 0:
        raise ValueError(
            f"gap_scale_days must be finite and positive; received {gap_scale_days}."
        )

    target_values = None
    if target is not None:
        target_values = np.asarray(target, dtype=np.float32).reshape(-1)
        if len(target_values) != n_time:
            raise ValueError(
                f"Target/SSM length mismatch: {len(target_values)} vs {n_time}."
            )

    if prepared_time_days is not None:
        if time_values is not None:
            raise ValueError(
                "Provide either time_values or prepared_time_days, not both."
            )
        time_days = np.asarray(prepared_time_days, dtype=np.float64).reshape(-1)
        if len(time_days) != n_time:
            raise ValueError(
                f"Prepared time/value length mismatch: {len(time_days)} vs {n_time}."
            )
        if not np.all(np.isfinite(time_days)):
            raise ValueError("Observation time coordinate contains non-finite values.")
        if len(time_days) > 1 and np.any(np.diff(time_days) <= 0):
            raise ValueError("Observation time coordinate must be strictly increasing.")
    else:
        time_days = _numeric_time_days(time_values, n_time)

    observation_indices = np.flatnonzero(np.isfinite(ssm))
    empty_dynamic = np.empty((0, window_size, 2), dtype=np.float32)
    empty_targets = None if target_values is None else np.empty((0, 1), dtype=np.float32)
    empty_indices = np.empty(0, dtype=np.int64)
    if len(observation_indices) < window_size:
        return empty_dynamic, empty_targets, empty_indices

    event_windows = np.lib.stride_tricks.sliding_window_view(
        observation_indices, window_size
    )
    candidate_indices = event_windows[:, -1]
    start = max(0, int(target_start))
    stop = n_time if target_stop is None else min(n_time, int(target_stop))
    if stop < start:
        raise ValueError(f"Invalid target interval [{start}, {stop}).")
    keep = (candidate_indices >= start) & (candidate_indices < stop)
    if target_values is not None:
        keep &= np.isfinite(target_values[candidate_indices])
    event_windows = event_windows[keep]
    candidate_indices = candidate_indices[keep]
    if len(candidate_indices) == 0:
        return empty_dynamic, empty_targets, empty_indices

    event_ssm = ssm[event_windows]
    event_times = time_days[event_windows]
    delta_t = np.diff(event_times, axis=1, prepend=event_times[:, :1])
    dynamic_columns = [
        event_ssm,
        delta_t.astype(np.float32) / np.float32(gap_scale_days),
    ]
    dynamic = np.ascontiguousarray(
        np.stack(dynamic_columns, axis=-1), dtype=np.float32
    )

    targets = None if target_values is None else target_values[candidate_indices, None].astype(np.float32)
    return dynamic, targets, candidate_indices.astype(np.int64)


def observation_event_support_mask(
    ssm,
    *,
    target_start=0,
    target_stop=None,
    window_size=OBSERVATION_EVENT_SEQUENCE_LENGTH,
    time_values=None,
    prepared_time_days=None,
):
    """Return finite-observation timestamps with a complete event history."""
    values = np.asarray(ssm)
    if values.ndim == 0:
        raise ValueError("SSM must have a time axis.")
    window_size = int(window_size)
    if window_size < 1:
        raise ValueError(f"window_size must be positive; received {window_size}.")
    n_time = values.shape[-1]
    if prepared_time_days is not None:
        if time_values is not None:
            raise ValueError(
                "Provide either time_values or prepared_time_days, not both."
            )
        time_days = np.asarray(prepared_time_days, dtype=np.float64).reshape(-1)
        if len(time_days) != n_time:
            raise ValueError(
                f"Prepared time/value length mismatch: {len(time_days)} vs {n_time}."
            )
    else:
        time_days = _numeric_time_days(time_values, n_time)
    if not np.all(np.isfinite(time_days)):
        raise ValueError("Observation time coordinate contains non-finite values.")
    if len(time_days) > 1 and np.any(np.diff(time_days) <= 0):
        raise ValueError("Observation time coordinate must be strictly increasing.")

    valid = np.isfinite(values)
    enough_events = np.cumsum(valid, axis=-1) >= window_size
    support = valid & enough_events
    start = max(0, int(target_start))
    stop = n_time if target_stop is None else min(n_time, int(target_stop))
    if stop < start:
        raise ValueError(f"Invalid target interval [{start}, {stop}).")
    interval = np.zeros(n_time, dtype=bool)
    interval[start:stop] = True
    return support & interval


def neural_development_metadata():
    """Identify the target-date split used for tuning and final training."""
    return {
        "validation_start_date": str(VALIDATION_START_DATE),
        "validation_end_date": str(MODEL_HOLDOUT_START_DATE - np.timedelta64(1, "D")),
        "model_holdout_start_date": str(MODEL_HOLDOUT_START_DATE),
    }


def ef_calibration_summary_path(project):
    """Return the canonical development-only EF calibration artifact path."""
    return os.path.join(
        project_results_root(project),
        "Exponential_Filter",
        "T_calibration",
        "EF_T_calibration_summary.json",
    )


def load_ef_t(project, product, *, required=True):
    """Return the EF estimation timescale without reading calibration files."""
    del project, required

    timescales = {
        "SMAP": 10.0,
        "ASCAT": 7.0,
    }

    product = str(product).upper()
    if product not in timescales:
        raise ValueError(f"Unsupported EF product: {product!r}")

    return timescales[product]


def neural_model_family(model_name):
    """Return ``FNO`` or ``LSTM`` for a supported neural scheme."""
    name = str(model_name).upper()
    if name in FNO_MODEL_CHOICES:
        return "FNO"
    if name in LSTM_MODEL_CHOICES:
        return "LSTM"
    raise ValueError(
        f"Unsupported neural model {model_name!r}; expected one of "
        f"{list(NEURAL_MODEL_CHOICES)}."
    )


def decode_netcdf_dates(time_values, units, calendar="standard"):
    """Decode a NetCDF time axis to day-resolution ``datetime64`` values."""

    if not units:
        raise ValueError("The NetCDF time variable is missing its units attribute.")
    decoded = nc.num2date(
        np.asarray(time_values),
        units=units,
        calendar=calendar,
        only_use_cftime_datetimes=False,
        only_use_python_datetimes=True,
    )
    return np.asarray(
        [np.datetime64(f"{d.year:04d}-{d.month:02d}-{d.day:02d}", "D") for d in decoded],
        dtype="datetime64[D]",
    )


def dates_from_variable(time_var):
    return decode_netcdf_dates(
        time_var[:],
        getattr(time_var, "units", None),
        getattr(time_var, "calendar", "standard"),
    )


def pixel_context_period_indices(
    dates,
    require_complete=True,
):
    """Indices of the daily context period and the target range within it."""
    dates = np.asarray(dates, dtype="datetime64[D]")
    indices = np.flatnonzero(
        (dates >= EF_INPUT_START_DATE) & (dates <= COMMON_END_DATE)
    )
    if len(indices) == 0:
        raise ValueError(
            f"No samples occur in the pixel context period "
            f"{EF_INPUT_START_DATE} to {COMMON_END_DATE}."
        )

    selected = dates[indices]
    if require_complete:
        expected = np.arange(
            EF_INPUT_START_DATE,
            COMMON_END_DATE + np.timedelta64(1, "D"),
            dtype="datetime64[D]",
        )
        if not np.array_equal(selected, expected):
            missing = expected[~np.isin(expected, selected)]
            duplicate_count = len(selected) - len(np.unique(selected))
            detail = []
            if len(missing):
                detail.append(f"missing {len(missing)} day(s), first={missing[0]}")
            if duplicate_count:
                detail.append(f"duplicate days={duplicate_count}")
            raise ValueError(
                "Pixel time axis is not the complete context daily calendar: "
                + (", ".join(detail) if detail else "unexpected ordering")
            )

    target_start_date = NEURAL_OUTPUT_START_DATE
    target_start = int(np.searchsorted(selected, target_start_date, side="left"))
    target_stop = int(np.searchsorted(
        selected,
        COMMON_END_DATE + np.timedelta64(1, "D"),
        side="left",
    ))
    if (
        target_start >= len(selected)
        or selected[target_start] != target_start_date
        or target_stop != len(selected)
    ):
        raise ValueError(
            f"Context calendar does not contain the complete target period "
            f"{target_start_date} to {COMMON_END_DATE}."
        )
    return indices, target_start, target_stop


def read_pixel_context_series(
    input_ds,
    *,
    ssm_variable,
    target_variable,
):
    """Read SSM and target of a pixel file over the daily context period."""
    required = ["time", ssm_variable, target_variable]
    missing = [name for name in required if name not in input_ds.variables]
    if missing:
        raise KeyError(f"Pixel NetCDF is missing variables: {', '.join(missing)}")

    dates_all = dates_from_variable(input_ds.variables["time"])
    context_idx, target_start, target_stop = pixel_context_period_indices(
        dates_all,
        require_complete=True,
    )

    def read_1d(name):
        values = np.ma.filled(input_ds.variables[name][:], np.nan)
        values = np.asarray(values, dtype=float).reshape(-1)
        if len(values) != len(dates_all):
            raise ValueError(
                f"Pixel variable {name!r} has {len(values)} samples but the "
                f"time coordinate has {len(dates_all)}."
            )
        return values[context_idx]

    dates = dates_all[context_idx]
    ssm = read_1d(ssm_variable)
    ssm[(ssm <= 0.0) | (ssm > 1.0)] = np.nan
    target = read_1d(target_variable)
    return ssm, target, dates, target_start, target_stop


def test_period_mask(dates):
    dates = np.asarray(dates, dtype="datetime64[D]")
    return (dates >= TEST_START_DATE) & (dates <= COMMON_END_DATE)


def test_period_metadata():
    """Return provenance fields for pixel metrics and figure inputs."""
    return {
        "test_period_policy": TEST_PERIOD_POLICY,
        "test_start_ratio_after_spinup": TEST_START_RATIO,
        "test_start_date": str(TEST_START_DATE),
        "test_end_date": str(COMMON_END_DATE),
    }


def split_boundaries(dates):
    """Return fitting/validation boundaries for the configured development period."""
    dates = np.asarray(dates, dtype="datetime64[D]")
    train_bound = int(np.searchsorted(dates, VALIDATION_START_DATE, side="left"))
    val_bound = int(np.searchsorted(dates, MODEL_HOLDOUT_START_DATE, side="left"))
    if train_bound <= 0 or val_bound <= train_bound or val_bound >= len(dates):
        raise ValueError(
            f"Invalid fixed split on {dates[0]}..{dates[-1]}: "
            f"train_bound={train_bound}, val_bound={val_bound}."
        )
    return train_bound, val_bound


def mask_to_ssm_observations(values, ssm, *, copy=True):
    """Set values to NaN where SSM is missing (``copy=False`` edits in place)."""
    if copy:
        masked = np.asarray(values, dtype=float).copy()
    else:
        masked = np.asarray(values)
        if not np.issubdtype(masked.dtype, np.floating):
            raise TypeError("In-place SSM support masking requires a floating-point array.")
        if not masked.flags.writeable:
            raise ValueError("In-place SSM support masking requires a writeable array.")
    ssm = np.asarray(ssm)
    if masked.shape != ssm.shape:
        raise ValueError(
            f"Value/SSM shape mismatch: {masked.shape} vs {ssm.shape}."
        )
    masked[~np.isfinite(ssm)] = np.nan
    return masked
