"""Coordinate-aligned daily rainfall and causal SMAP-event accumulation.

Rainfall is GPM IMERG summed over local days from 06:00 to 06:00 (local =
UTC + round(lon/15) h; GPM_preprocessing_daily_local_totals.py --day-start-hour 6).
For consecutive valid SMAP dates a < b, the event value at b is the sum of
the daily totals on dates a through b-1, i.e. the rain between the 06:00 SMAP
passes on a and b. It is log1p-transformed and z-scored with the training-only
scaler; the first event of each window is set to zero.
"""

from pathlib import Path

import netCDF4 as nc
import numpy as np
import torch

from core.project_paths import project_results_root

EVENT_PRECIPITATION_CHANNELS = [
    'SSM', 'delta_t_days_scaled', 'log1p_precipitation_between_events_standardized',
]


RAINFALL_FILE = 'GPM_precipitation_2015_2023_local_06-06.nc'
RAINFALL_VARIABLE = 'daily_precipitation_sum'


def rainfall_path(project):
    """Locate the daily rainfall source used to prepare both neural families."""
    return project_results_root(project) / 'Global_Precipitation_Measurement' / RAINFALL_FILE


def require_rainfall_file(name, where):
    """Stop unless ``name`` (a file name) is the 06:00-06:00 local-day rain file."""
    if name != RAINFALL_FILE:
        raise ValueError(f'{where}: rainfall comes from {name or "an unrecorded file"!r}, '
                         f'not {RAINFALL_FILE!r}; rebuild it from the 06:00-06:00 rain')


def daily_dates(variable):
    calendar = getattr(variable, "calendar", "standard")
    if calendar not in ("standard", "gregorian", "proleptic_gregorian"):
        raise ValueError(f"Unsupported rainfall calendar: {calendar}")
    decoded = nc.num2date(variable[:], variable.units, calendar=calendar)
    if any((date.hour, date.minute, date.second) != (0, 0, 0) for date in decoded):
        raise ValueError("Rainfall requires one midnight-labelled daily total per date")
    dates = np.array([f"{d.year:04d}-{d.month:02d}-{d.day:02d}" for d in decoded], dtype="datetime64[D]")
    if len(dates) < 2 or not np.all(np.diff(dates) == np.timedelta64(1, "D")):
        raise ValueError("Rainfall dates must be unique, ordered, and continuous daily samples")
    return dates


def coordinate_indices(axis, requested, name):
    axis = np.ma.asarray(axis, dtype=float).filled(np.nan)
    requested = np.asarray(requested, dtype=float)
    if axis.ndim != 1 or not np.isfinite(axis).all() or len(np.unique(axis)) != len(axis):
        raise ValueError(f"Invalid rainfall {name} coordinates")
    order = np.argsort(axis)
    values = axis[order]
    positions = np.searchsorted(values, requested)
    left = np.clip(positions - 1, 0, len(values) - 1)
    right = np.clip(positions, 0, len(values) - 1)
    chosen = np.where(abs(values[left] - requested) <= abs(values[right] - requested), left, right)
    indices = order[chosen]
    if not np.allclose(axis[indices], requested, atol=1e-5, rtol=0):
        raise ValueError(f"Rainfall {name} grid must match training coordinates; regrid the source first")
    return indices


def extract_daily_pixels(path, variable_name, table, start, end, report_progress=True):
    """Read paired pixels by spatial/time chunks, not the full global cube."""
    with nc.Dataset(path) as ds:
        dates = daily_dates(ds["time"])
        requested_dates = np.arange(np.datetime64(start, "D"),
                                    np.datetime64(end, "D") + np.timedelta64(1, "D"),
                                    dtype="datetime64[D]")
        positions = np.searchsorted(dates, requested_dates)
        if ((positions >= len(dates)).any() or not np.array_equal(dates[positions], requested_dates)):
            raise ValueError(f"Rainfall file must cover {start} through {end}")
        var = ds[variable_name]
        if len(var.dimensions) != 3 or set(var.dimensions) != {'time', 'lat', 'lon'}:
            raise ValueError('Rainfall must have time, lat, lon dimensions in any order')
        axes = {name: var.dimensions.index(name) for name in ('time', 'lat', 'lon')}
        order = tuple(axes[name] for name in ('time', 'lat', 'lon'))
        units = str(getattr(var, "units", "")).lower().replace(" ", "")
        if units not in ("mm/day", "mmd-1", "mmday-1", "mm"):
            raise ValueError(f"Expected daily precipitation totals in mm or mm/day; received {units!r}")
        lat = coordinate_indices(ds["lat"][:], table.lat.to_numpy(), "latitude")
        lon = coordinate_indices(ds["lon"][:], table.lon.to_numpy(), "longitude")
        chunks = var.chunking()
        chunks = (32, 32, 64) if chunks == "contiguous" else tuple(chunks[axis] for axis in order)
        groups = {}
        for i, (y, x) in enumerate(zip(lat, lon)):
            groups.setdefault((int(y // chunks[1]), int(x // chunks[2])), []).append(i)
        output = np.full((len(table), len(positions)), np.nan, dtype=np.float32)
        first, stop = int(positions[0]), int(positions[-1]) + 1
        for number, ((by, bx), pixels) in enumerate(sorted(groups.items()), 1):
            y0, x0 = by * chunks[1], bx * chunks[2]
            ys = slice(y0, min(y0 + chunks[1], var.shape[axes["lat"]]))
            xs = slice(x0, min(x0 + chunks[2], var.shape[axes["lon"]]))
            for block in range(first // chunks[0], (stop - 1) // chunks[0] + 1):
                lo, hi = max(first, block * chunks[0]), min(stop, (block + 1) * chunks[0])
                selections = [slice(None)] * 3
                selections[axes['time']] = slice(lo, hi)
                selections[axes['lat']] = ys
                selections[axes['lon']] = xs
                cube = np.ma.asarray(var[tuple(selections)], dtype=float).filled(np.nan)
                cube = np.transpose(cube, order)
                output[pixels, lo - first:hi - first] = cube[:, lat[pixels] - y0, lon[pixels] - x0].T
            if report_progress:
                print(f"Rainfall extraction: spatial chunk {number}/{len(groups)}", flush=True)
        return requested_dates, output, {
            "variable": variable_name, "units": str(var.units), "dimensions": list(var.dimensions),
            "source_start_date": str(dates[0]), "source_end_date": str(dates[-1]),
            "extracted_start_date": str(requested_dates[0]), "extracted_end_date": str(requested_dates[-1]),
            "coordinate_matching": "exact lat/lon within 1e-5 degrees; no raster-index reuse or interpolation",
        }


def event_interval_amounts(ssm, dates, rain_dates, daily_mm, lag_days=1):
    """For events a,b, use totals on dates [a-lag+1, b-lag], inclusive."""
    if not isinstance(lag_days, int) or isinstance(lag_days, bool) or lag_days < 1:
        raise ValueError("Rainfall lag must be at least one completed calendar day")
    dates = np.asarray(dates, dtype="datetime64[D]")
    rain_dates = np.asarray(rain_dates, dtype="datetime64[D]")
    values = np.ma.asarray(daily_mm, dtype=float).filled(np.nan)
    if (values.ndim != 1 or len(values) != len(rain_dates) or not len(rain_dates)
            or not np.all(np.diff(rain_dates) == np.timedelta64(1, "D"))):
        raise ValueError("Precipitation must have a continuous daily time axis and one total per date")
    ssm = np.asarray(ssm)
    if len(ssm) != len(dates) or not np.all(np.diff(dates) == np.timedelta64(1, "D")):
        raise ValueError("SMAP context must have the matching complete daily calendar")
    valid = np.isfinite(values) & (values >= 0)
    total = np.r_[0., np.cumsum(np.where(valid, values, 0.), dtype=np.float64)]
    missing = np.r_[0, np.cumsum(~valid)]
    events = np.flatnonzero(np.isfinite(ssm))
    result = np.full(len(dates), np.nan, dtype=np.float32)
    if not len(events):
        return result
    result[events[0]] = 0.
    left = dates[events[:-1]] - np.timedelta64(lag_days - 1, "D")
    right = dates[events[1:]] - np.timedelta64(lag_days, "D")
    lo = np.searchsorted(rain_dates, left)
    hi = np.searchsorted(rain_dates, right, side="right")
    covered = (left >= rain_dates[0]) & (right <= rain_dates[-1]) & (hi > lo)
    covered &= (hi - lo) == ((right - left) / np.timedelta64(1, "D") + 1)
    covered &= missing[hi] == missing[lo]
    result[events[1:][covered]] = (total[hi[covered]] - total[lo[covered]]).astype(np.float32)
    return result


def event_windows(ssm, target_indices, window_size=32):
    events = np.flatnonzero(np.isfinite(ssm))
    indices = np.asarray(target_indices, dtype=np.int64)
    if not len(indices):
        return np.empty((0, window_size), dtype=np.int64)
    positions = np.searchsorted(events, indices)
    if (positions >= len(events)).any() or (positions < window_size - 1).any() or not np.array_equal(events[positions], indices):
        raise ValueError("Targets must be valid SMAP events with a complete event history")
    return events[positions[:, None] - np.arange(window_size - 1, -1, -1)[None, :]]


def precipitation_windows(ssm, target_indices, interval_mm, window_size=32):
    indices = event_windows(ssm, target_indices, window_size)
    values = np.asarray(interval_mm)[indices].copy()
    values[:, 0] = 0.
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Missing/negative precipitation in required SMAP history; no windows may be silently dropped or imputed")
    return values


def load_interval_series(path, dates):
    dates = np.asarray(dates, dtype="datetime64[D]")
    with np.load(Path(path), allow_pickle=False) as data:
        saved_dates, values = data["dates"], data["interval_mm"]
    positions = np.searchsorted(saved_dates, dates)
    in_range = positions < len(saved_dates)
    in_range[in_range] &= saved_dates[positions[in_range]] == dates[in_range]
    output = np.full(len(dates), np.nan, dtype=np.float32)
    output[in_range] = values[positions[in_range]]
    return output


def standardized_precipitation_windows(ssm, targets, interval_mm, scaler, window_size=32):
    """Shared training/pixel/CONUS N=1 transform; first context entry is zero."""
    raw = precipitation_windows(ssm, targets, interval_mm, window_size)
    if not np.isfinite(scaler["mean"]) or not np.isfinite(scaler["std"]) or scaler["std"] <= 0:
        raise ValueError("Invalid training-only precipitation scaler")
    values = (np.log1p(raw.astype(np.float64)) - scaler["mean"]) / scaler["std"]
    values[:, 0] = 0.
    return values.astype(np.float32)


def append_precipitation_channel(dataset, ssm, interval_mm, scaler, *, enabled):
    values = standardized_precipitation_windows(
        ssm, dataset.target_indices, interval_mm, scaler, dataset.window_size,
    )
    if not enabled:
        values.fill(0.)
    extra = torch.tensor(values[..., None], dtype=torch.float32)
    dataset.X_dyn = torch.cat((dataset.X_dyn, extra), dim=-1)
