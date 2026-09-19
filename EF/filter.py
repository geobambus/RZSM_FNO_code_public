"""Shared observation-time exponential filter for point and CONUS workflows."""

from __future__ import annotations

import numpy as np
import pandas as pd


def elapsed_days(time_values):
    """Convert a strictly increasing one-dimensional time coordinate to days."""
    dates = pd.DatetimeIndex(pd.to_datetime(np.asarray(time_values).reshape(-1)))
    if dates.empty:
        return np.empty(0, dtype=float)
    if dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("Time coordinates must be unique and strictly increasing.")
    return ((dates - dates[0]) / pd.Timedelta(days=1)).to_numpy(dtype=float)


def exponential_filter(ssm, time_days, time_constant_days):
    """Filter one or more SSM series, updating only at finite observations.

    ``ssm`` may have shape ``(time,)`` or ``(..., time)``. Missing retrievals
    remain missing in the returned array, while the internal state continues
    across observation gaps.
    """
    values = np.asarray(ssm, dtype=np.float32)
    original_shape = values.shape
    if values.ndim == 1:
        values = values[np.newaxis, :]
    elif values.ndim < 1:
        raise ValueError("SSM must include a time dimension.")
    times = np.asarray(time_days, dtype=np.float64).reshape(-1)
    if values.shape[-1] != len(times):
        raise ValueError("The final SSM dimension must match the time coordinate.")
    if len(times) > 1 and np.any(np.diff(times) <= 0):
        raise ValueError("Observation dates must be strictly increasing.")
    if not np.isfinite(time_constant_days) or float(time_constant_days) <= 0:
        raise ValueError("The EF time constant must be finite and positive.")

    flat = values.reshape(-1, values.shape[-1]).copy()
    flat[(flat <= 0.0) | (flat > 1.0)] = np.nan
    prediction = np.full_like(flat, np.nan)
    state = np.full(flat.shape[0], np.nan, dtype=np.float32)
    previous_gain = np.ones(flat.shape[0], dtype=np.float32)
    previous_time = np.zeros(flat.shape[0], dtype=np.float64)
    initialized = np.zeros(flat.shape[0], dtype=bool)

    for time_index, day in enumerate(times):
        valid = np.isfinite(flat[:, time_index])
        new = valid & ~initialized
        state[new] = flat[new, time_index]
        previous_gain[new] = 1.0
        previous_time[new] = day
        initialized[new] = True

        update = valid & initialized & ~new
        if np.any(update):
            decay = np.exp(
                -(day - previous_time[update]) / float(time_constant_days)
            ).astype(np.float32)
            gain = previous_gain[update] / (previous_gain[update] + decay)
            state[update] += gain * (flat[update, time_index] - state[update])
            previous_gain[update] = gain
            previous_time[update] = day
        prediction[valid, time_index] = state[valid]

    return prediction.reshape(original_shape)


def filtered_rzsm(ssm, time_values, time_constant_days, spin_up_days=365.0):
    """Calculate EF RZSM and mask the elapsed-day adjustment period."""
    days = elapsed_days(time_values)
    prediction = exponential_filter(ssm, days, time_constant_days)
    prediction = np.asarray(prediction, dtype=float)
    prediction[..., days < float(spin_up_days)] = np.nan
    return prediction
