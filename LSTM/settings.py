"""Scientific constants and sequence construction shared by LSTM workflows."""

import numpy as np


MODEL_SEEDS = (42, 111, 222, 333, 444, 555, 666, 777, 888, 999)
OPTUNA_SEED = MODEL_SEEDS[0]

PRODUCTS = ("ASCAT", "SMAP")
AREAS = ("Train", "Test", "Excluded")
PRODUCT_SETTINGS = {
    "ASCAT": {"folder": "ASCAT", "ssm_variable": "ASCAT_SSM"},
    "SMAP": {"folder": "SMAP", "ssm_variable": "SMAP_SSM"},
}
TARGET_VARIABLE = "in-situ_RZSM"

MODEL_INPUT_METADATA_COLUMNS = (
    "lat_idx",
    "lon_idx",
    "lat",
    "lon",
    "valid_SSM",
    "valid_RZSM",
)
STATIC_FEATURES = (
    "LST_avg",
    "NDVI_avg",
    "Bulk_density_RZSM",
    "Sand_RZSM",
    "Clay_RZSM",
    "Water_content_10kPa_RZSM",
    "Water_content_1500kPa_RZSM",
)

CONTEXT_START_DATE = np.datetime64("2015-04-01", "D")
CONTEXT_END_DATE = np.datetime64("2023-12-31", "D")
VALIDATION_START_DATE = np.datetime64("2021-06-15", "D")
MODEL_HOLDOUT_START_DATE = np.datetime64("2022-04-21", "D")
WINDOW_SIZE = 32
OBSERVATION_GAP_SCALE_DAYS = float(WINDOW_SIZE)
SPATIAL_SPLIT_SEED = 42
SPATIAL_TRAIN_FRACTION = 0.80


def validate_product(product):
    """Return an uppercase product name or raise a clear error."""
    product = str(product).upper()
    if product not in PRODUCTS:
        raise ValueError(f"Unsupported product {product!r}; expected {PRODUCTS}.")
    return product


def validate_static_columns(columns, source_label="model input"):
    """Require the exact ordered seven-feature static schema."""
    columns = list(columns)
    actual = [
        name for name in columns if name not in MODEL_INPUT_METADATA_COLUMNS
    ]
    expected = list(STATIC_FEATURES)
    if actual != expected:
        missing = [name for name in expected if name not in actual]
        extra = [name for name in actual if name not in expected]
        raise ValueError(
            f"{source_label} does not contain the required ordered features. "
            f"Expected {expected}; received {actual}. "
            f"Missing={missing}, extra={extra}."
        )
    return expected


def dates_from_variable(time_variable):
    """Decode a NetCDF time coordinate to day-resolution datetime64 values."""
    import netCDF4 as nc

    units = getattr(time_variable, "units", None)
    if not units:
        raise ValueError("The NetCDF time variable has no units attribute.")
    decoded = nc.num2date(
        time_variable[:],
        units=units,
        calendar=getattr(time_variable, "calendar", "standard"),
        only_use_cftime_datetimes=False,
        only_use_python_datetimes=True,
    )
    return np.asarray(
        [
            np.datetime64(
                f"{value.year:04d}-{value.month:02d}-{value.day:02d}", "D"
            )
            for value in decoded
        ],
        dtype="datetime64[D]",
    )


def context_period_indices(dates, require_complete=True):
    """Select and validate the fixed 2015-04-01 through 2023-12-31 calendar."""
    dates = np.asarray(dates, dtype="datetime64[D]")
    indices = np.flatnonzero(
        (dates >= CONTEXT_START_DATE) & (dates <= CONTEXT_END_DATE)
    )
    if len(indices) == 0:
        raise ValueError("No samples occur in the point-model context period.")
    selected = dates[indices]
    if require_complete:
        expected = np.arange(
            CONTEXT_START_DATE,
            CONTEXT_END_DATE + np.timedelta64(1, "D"),
            dtype="datetime64[D]",
        )
        if not np.array_equal(selected, expected):
            raise ValueError(
                "Point time axis is not the complete daily calendar from "
                f"{CONTEXT_START_DATE} through {CONTEXT_END_DATE}."
            )
    return indices


def read_point_series(dataset, product, target_variable=TARGET_VARIABLE):
    """Read physical SSM and observed RZSM on the fixed point-model calendar."""
    product = validate_product(product)
    ssm_variable = PRODUCT_SETTINGS[product]["ssm_variable"]
    required = ("time", ssm_variable, target_variable)
    missing = [name for name in required if name not in dataset.variables]
    if missing:
        raise KeyError(f"Point NetCDF is missing variables: {missing}")

    all_dates = dates_from_variable(dataset.variables["time"])
    indices = context_period_indices(all_dates, require_complete=True)

    def read_one(name):
        values = np.ma.filled(dataset.variables[name][:], np.nan)
        values = np.asarray(values, dtype=float).reshape(-1)
        if len(values) != len(all_dates):
            raise ValueError(
                f"{name!r} has {len(values)} values for {len(all_dates)} dates."
            )
        return values[indices]

    ssm = read_one(ssm_variable)
    ssm[(ssm <= 0.0) | (ssm > 1.0)] = np.nan
    target = read_one(target_variable)
    return ssm, target, all_dates[indices], indices


def split_boundaries(dates):
    """Return the fixed temporal training/validation/holdout boundaries."""
    dates = np.asarray(dates, dtype="datetime64[D]")
    train_stop = int(np.searchsorted(dates, VALIDATION_START_DATE, side="left"))
    validation_stop = int(
        np.searchsorted(dates, MODEL_HOLDOUT_START_DATE, side="left")
    )
    if not 0 < train_stop < validation_stop < len(dates):
        raise ValueError(
            f"Invalid temporal split for {dates[0]} through {dates[-1]}."
        )
    return train_stop, validation_stop


def _numeric_time_days(time_values, length):
    """Convert a strictly increasing time coordinate to elapsed days."""
    if time_values is None:
        values = np.arange(length, dtype=np.float64)
    else:
        values = np.asarray(time_values)
        if len(values) != length:
            raise ValueError(f"Time/value length mismatch: {len(values)} vs {length}.")
        if np.issubdtype(values.dtype, np.datetime64):
            values = (
                values.astype("datetime64[D]") - values[0].astype("datetime64[D]")
            ) / np.timedelta64(1, "D")
        values = values.astype(np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("Observation time coordinate contains non-finite values.")
    if len(values) > 1 and np.any(np.diff(values) <= 0):
        raise ValueError("Observation time coordinate must be strictly increasing.")
    return values


def build_observation_sequences(
    ssm,
    target=None,
    *,
    time_values=None,
    target_start=0,
    target_stop=None,
    window_size=WINDOW_SIZE,
):
    """Build past-only windows from the last finite satellite observations."""
    ssm = np.asarray(ssm, dtype=np.float32).reshape(-1)
    length = len(ssm)
    target_values = None
    if target is not None:
        target_values = np.asarray(target, dtype=np.float32).reshape(-1)
        if len(target_values) != length:
            raise ValueError("Target and SSM lengths differ.")

    observation_indices = np.flatnonzero(np.isfinite(ssm))
    empty_dynamic = np.empty((0, window_size, 2), dtype=np.float32)
    empty_target = (
        None if target_values is None else np.empty((0, 1), dtype=np.float32)
    )
    empty_indices = np.empty(0, dtype=np.int64)
    if len(observation_indices) < window_size:
        return empty_dynamic, empty_target, empty_indices

    time_days = _numeric_time_days(time_values, length)
    observation_times = time_days[observation_indices]
    gaps = np.zeros(len(observation_indices), dtype=np.float32)
    gaps[1:] = np.diff(observation_times).astype(np.float32)
    events = np.stack(
        (ssm[observation_indices], gaps / OBSERVATION_GAP_SCALE_DAYS), axis=1
    ).astype(np.float32)

    start = max(0, int(target_start))
    stop = length if target_stop is None else min(length, int(target_stop))
    end_positions = np.arange(window_size - 1, len(observation_indices))
    candidate_indices = observation_indices[end_positions]
    keep = (candidate_indices >= start) & (candidate_indices < stop)
    if target_values is not None:
        keep &= np.isfinite(target_values[candidate_indices])
    end_positions = end_positions[keep]
    candidate_indices = candidate_indices[keep]
    if len(end_positions) == 0:
        return empty_dynamic, empty_target, empty_indices

    dynamic = np.stack(
        [events[end - window_size + 1 : end + 1] for end in end_positions]
    ).astype(np.float32)
    targets = (
        None
        if target_values is None
        else target_values[candidate_indices, None].astype(np.float32)
    )
    return dynamic, targets, candidate_indices.astype(np.int64)


def summarize_ensemble(predictions):
    """Return per-timestep ensemble mean and sample standard deviation."""
    values = np.asarray(predictions, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("Ensemble predictions must have shape (seed, time).")
    finite = np.isfinite(values)
    counts = finite.sum(axis=0)
    sums = np.where(finite, values, 0.0).sum(axis=0)
    mean = np.full(values.shape[1], np.nan, dtype=np.float64)
    np.divide(sums, counts, out=mean, where=counts > 0)
    centered = np.where(finite, values - mean, 0.0)
    sum_squared = np.square(centered).sum(axis=0)
    variance = np.full(values.shape[1], np.nan, dtype=np.float64)
    np.divide(sum_squared, counts - 1, out=variance, where=counts > 1)
    return mean, np.sqrt(variance)
