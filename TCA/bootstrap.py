"""Memory-bounded bootstrap ETC with compact final-only NetCDF outputs."""

from __future__ import annotations

import hashlib
import math
import multiprocessing as mp
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import netCDF4 as nc
import numpy as np
from tqdm.auto import tqdm


@dataclass(frozen=True)
class ProductSpec:
    """One gridded RZSM input used by the TCA calculation."""

    path: Path | str
    variable: str
    stem: str


@dataclass(frozen=True)
class Triplet:
    """Ordered ASCAT, SMAP, and land-reference products."""

    ascat: str
    smap: str
    reference: str
    name: str


LAND_REFERENCES = (
    "ERA5-Land",
    "NLDAS_NOAH",
    "NLDAS_VIC",
    "NLDAS_MOSAIC",
)

FIXED_TRIPLETS = tuple(
    Triplet(ascat, smap, reference, f"{ascat.removeprefix('ASCAT_')}_{smap.removeprefix('SMAP_')}_{reference}")
    for ascat, smap in (
        ("ASCAT_EF", "SMAP_EF"),
        ("ASCAT_FNO", "SMAP_EF"),
        ("ASCAT_LSTM", "SMAP_EF"),
        ("ASCAT_EF", "SMAP_FNO"),
        ("ASCAT_EF", "SMAP_LSTM"),
    )
    for reference in LAND_REFERENCES
)


_WORKER_CUBES = None
_WORKER_INDICES = None
_WORKER_NOD_THRESHOLD = None
_WORKER_CORRELATION_THRESHOLD = None


def _filled(values, dtype=np.float32):
    return np.asarray(np.ma.filled(values, np.nan), dtype=dtype)


def _coordinate_axes(dataset):
    for latitude_name, longitude_name in (
        ("lat", "lon"),
        ("latitude", "longitude"),
    ):
        if (
            latitude_name in dataset.variables
            and longitude_name in dataset.variables
        ):
            latitude = _filled(dataset.variables[latitude_name][:])
            longitude = _filled(dataset.variables[longitude_name][:])
            if latitude.ndim == longitude.ndim == 1:
                return latitude, longitude
    if "x" not in dataset.variables or "y" not in dataset.variables:
        raise KeyError("Could not find regular latitude and longitude coordinates.")
    x = _filled(dataset.variables["x"][:])
    y = _filled(dataset.variables["y"][:])
    if x.ndim != 2 or y.ndim != 2 or x.shape != y.shape:
        raise ValueError("x/y coordinates must be matching regular 2-D grids.")
    longitude = x[0]
    latitude = y[:, 0]
    if not np.allclose(x, longitude[None, :], atol=1e-4, equal_nan=True):
        raise ValueError("x is not a regular longitude grid.")
    if not np.allclose(y, latitude[:, None], atol=1e-4, equal_nan=True):
        raise ValueError("y is not a regular latitude grid.")
    return latitude, longitude


def _dates(dataset):
    """Decode input dates, including xarray-style nanosecond coordinates."""
    if "time" not in dataset.variables:
        raise KeyError("TCA input has no time coordinate.")
    variable = dataset.variables["time"]
    units = getattr(variable, "units", None)
    if not units:
        raise ValueError("TCA input time coordinate has no units.")
    values = variable[:]
    unit_name, separator, reference_time = units.partition(" since ")
    if separator and unit_name.strip().lower() in {"nanosecond", "nanoseconds"}:
        # netCDF4/cftime does not recognize nanoseconds, but xarray may write
        # datetime64[ns] coordinates using them. Microseconds are supported and
        # preserve far more precision than this workflow's daily comparison.
        values = np.ma.asarray(values, dtype=np.float64) / 1_000.0
        units = f"microseconds since {reference_time}"
    decoded = nc.num2date(
        values,
        units=units,
        calendar=getattr(variable, "calendar", "standard"),
        only_use_cftime_datetimes=False,
        only_use_python_datetimes=True,
    )
    result = np.asarray(
        [
            np.datetime64(
                f"{value.year:04d}-{value.month:02d}-{value.day:02d}", "D"
            )
            for value in decoded
        ],
        dtype="datetime64[D]",
    )
    if len(result) > 1 and np.any(np.diff(result) <= np.timedelta64(0, "D")):
        raise ValueError("TCA input dates must be unique and increasing.")
    return result


def _regional_slice(latitude, longitude, bounds):
    lon_min, lon_max, lat_min, lat_max = (float(value) for value in bounds)
    latitude_indices = np.flatnonzero(
        (latitude >= lat_min) & (latitude <= lat_max)
    )
    longitude_indices = np.flatnonzero(
        (longitude >= lon_min) & (longitude <= lon_max)
    )
    if not len(latitude_indices) or not len(longitude_indices):
        raise ValueError(f"No TCA input cells occur inside bounds {bounds}.")
    return (
        slice(int(latitude_indices.min()), int(latitude_indices.max()) + 1),
        slice(int(longitude_indices.min()), int(longitude_indices.max()) + 1),
    )


def _axis_flip(reference, candidate, label, tolerance=1e-4):
    if reference.shape != candidate.shape:
        raise ValueError(
            f"{label} shapes differ: {reference.shape} and {candidate.shape}."
        )
    if np.allclose(reference, candidate, atol=tolerance, rtol=0.0):
        return False
    if np.allclose(reference, candidate[::-1], atol=tolerance, rtol=0.0):
        return True
    raise ValueError(f"{label} coordinates are not aligned.")


def _spatial_dimensions(variable, latitude_length, longitude_length):
    dimensions = list(variable.dimensions)
    if "time" not in dimensions:
        raise ValueError(f"{variable.name!r} has no time dimension.")
    candidates = [name for name in dimensions if name != "time"]
    latitude_names = [
        name
        for name in candidates
        if name.lower() in {"lat", "latitude", "y"}
        and len(variable.group().dimensions[name]) == latitude_length
    ]
    longitude_names = [
        name
        for name in candidates
        if name.lower() in {"lon", "longitude", "x"}
        and len(variable.group().dimensions[name]) == longitude_length
    ]
    if not latitude_names:
        latitude_names = [
            name
            for name in candidates
            if len(variable.group().dimensions[name]) == latitude_length
        ]
    if not longitude_names:
        longitude_names = [
            name
            for name in candidates
            if len(variable.group().dimensions[name]) == longitude_length
            and name not in latitude_names[:1]
        ]
    if not latitude_names or not longitude_names:
        raise ValueError(
            f"Could not infer spatial dimensions for {variable.name!r}."
        )
    return latitude_names[0], longitude_names[0]


def _read_chunk(
    variable,
    latitude_dimension,
    longitude_dimension,
    latitude_slice,
    longitude_slice,
    time_indices,
):
    selections = []
    axes = {}
    for dimension in variable.dimensions:
        if dimension == latitude_dimension:
            axes["lat"] = len(selections)
            selections.append(latitude_slice)
        elif dimension == longitude_dimension:
            axes["lon"] = len(selections)
            selections.append(longitude_slice)
        elif dimension == "time":
            axes["time"] = len(selections)
            if len(time_indices) and np.all(np.diff(time_indices) == 1):
                selections.append(
                    slice(int(time_indices[0]), int(time_indices[-1]) + 1)
                )
            else:
                selections.append(time_indices)
        else:
            selections.append(0)
    values = _filled(variable[tuple(selections)])
    return np.transpose(values, (axes["lat"], axes["lon"], axes["time"]))


def _rolling_anomaly(values, window_size):
    values = np.asarray(values, dtype=np.float32)
    time_count = values.shape[-1]
    half_window = int(window_size) // 2
    starts = np.maximum(np.arange(time_count) - half_window, 0)
    stops = np.minimum(np.arange(time_count) + half_window + 1, time_count)
    valid = np.isfinite(values)
    sums = np.cumsum(
        np.where(valid, values, 0.0), axis=-1, dtype=np.float64
    )
    counts = np.cumsum(valid, axis=-1, dtype=np.int32)
    sums = np.concatenate(
        (np.zeros((*sums.shape[:-1], 1), dtype=np.float64), sums), axis=-1
    )
    counts = np.concatenate(
        (np.zeros((*counts.shape[:-1], 1), dtype=np.int32), counts), axis=-1
    )
    window_sums = sums[..., stops] - sums[..., starts]
    window_counts = counts[..., stops] - counts[..., starts]
    mean = np.full(values.shape, np.nan, dtype=np.float32)
    np.divide(window_sums, window_counts, out=mean, where=window_counts > 0)
    return values - mean


def _stage_anomaly(
    spec,
    target_latitude,
    target_longitude,
    expected_dates,
    bounds,
    output_path,
    window_size,
    chunk_latitude,
    chunk_longitude,
):
    path = Path(spec.path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with nc.Dataset(path) as dataset:
        if spec.variable not in dataset.variables:
            raise KeyError(f"{path} has no {spec.variable!r} variable.")
        latitude, longitude = _coordinate_axes(dataset)
        latitude_slice, longitude_slice = _regional_slice(
            latitude, longitude, bounds
        )
        regional_latitude = latitude[latitude_slice]
        regional_longitude = longitude[longitude_slice]
        flip_latitude = _axis_flip(
            target_latitude, regional_latitude, f"{path.name} latitude"
        )
        flip_longitude = _axis_flip(
            target_longitude, regional_longitude, f"{path.name} longitude"
        )
        source_dates = _dates(dataset)
        date_lookup = {value: index for index, value in enumerate(source_dates)}
        missing_dates = [value for value in expected_dates if value not in date_lookup]
        if missing_dates:
            raise ValueError(
                f"{path} is missing {len(missing_dates)} comparison dates; "
                f"first missing date is {missing_dates[0]}."
            )
        time_indices = np.asarray(
            [date_lookup[value] for value in expected_dates], dtype=np.int64
        )
        variable = dataset.variables[spec.variable]
        latitude_dimension, longitude_dimension = _spatial_dimensions(
            variable, len(latitude), len(longitude)
        )

        output = np.lib.format.open_memmap(
            output_path,
            mode="w+",
            dtype=np.float32,
            shape=(
                len(target_latitude),
                len(target_longitude),
                len(expected_dates),
            ),
        )
        try:
            latitude_count = len(target_latitude)
            longitude_count = len(target_longitude)
            for i0 in range(0, latitude_count, int(chunk_latitude)):
                i1 = min(i0 + int(chunk_latitude), latitude_count)
                source_i0, source_i1 = (
                    (latitude_count - i1, latitude_count - i0)
                    if flip_latitude
                    else (i0, i1)
                )
                for j0 in range(0, longitude_count, int(chunk_longitude)):
                    j1 = min(j0 + int(chunk_longitude), longitude_count)
                    source_j0, source_j1 = (
                        (longitude_count - j1, longitude_count - j0)
                        if flip_longitude
                        else (j0, j1)
                    )
                    chunk = _read_chunk(
                        variable,
                        latitude_dimension,
                        longitude_dimension,
                        slice(
                            latitude_slice.start + source_i0,
                            latitude_slice.start + source_i1,
                        ),
                        slice(
                            longitude_slice.start + source_j0,
                            longitude_slice.start + source_j1,
                        ),
                        time_indices,
                    )
                    if flip_latitude:
                        chunk = chunk[::-1]
                    if flip_longitude:
                        chunk = chunk[:, ::-1]
                    output[i0:i1, j0:j1] = _rolling_anomaly(
                        chunk, window_size
                    )
            output.flush()
        finally:
            del output
    return Path(output_path)


def _initialize_worker(cube_paths, bootstrap_indices_path, nod_th, corr_th):
    global _WORKER_CUBES, _WORKER_INDICES
    global _WORKER_NOD_THRESHOLD, _WORKER_CORRELATION_THRESHOLD
    _WORKER_CUBES = tuple(np.load(path, mmap_mode="r") for path in cube_paths)
    _WORKER_INDICES = np.load(bootstrap_indices_path, mmap_mode="r")
    _WORKER_NOD_THRESHOLD = int(nod_th)
    _WORKER_CORRELATION_THRESHOLD = float(corr_th)


def _covariance_correlation_three(x, y, z):
    """Return pairwise covariance and correlation grids along the last axis."""
    valid = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
    x_valid = np.where(valid, x, np.nan)
    y_valid = np.where(valid, y, np.nan)
    z_valid = np.where(valid, z, np.nan)
    count = valid.sum(axis=-1)

    def covariance(left, right):
        left_centered = left - np.nanmean(left, axis=-1, keepdims=True)
        right_centered = right - np.nanmean(right, axis=-1, keepdims=True)
        return np.nansum(left_centered * right_centered, axis=-1) / (count - 1)

    cov_xx = covariance(x_valid, x_valid)
    cov_yy = covariance(y_valid, y_valid)
    cov_zz = covariance(z_valid, z_valid)
    cov_xy = covariance(x_valid, y_valid)
    cov_xz = covariance(x_valid, z_valid)
    cov_yz = covariance(y_valid, z_valid)
    return {
        "covXX": cov_xx,
        "covYY": cov_yy,
        "covZZ": cov_zz,
        "covXY": cov_xy,
        "covXZ": cov_xz,
        "covYZ": cov_yz,
        "corrXY": cov_xy / np.sqrt(cov_xx * cov_yy),
        "corrXZ": cov_xz / np.sqrt(cov_xx * cov_zz),
        "corrYZ": cov_yz / np.sqrt(cov_yy * cov_zz),
    }


def _extended_triple_collocation(x, y, z, nod_th=30, corr_th=0.0):
    """Calculate vectorized extended triple-collocation diagnostics."""
    statistics = _covariance_correlation_three(x, y, z)
    cov_xx = statistics["covXX"]
    cov_yy = statistics["covYY"]
    cov_zz = statistics["covZZ"]
    cov_xy = statistics["covXY"]
    cov_xz = statistics["covXZ"]
    cov_yz = statistics["covYZ"]

    var_error = {
        "x": cov_xx - cov_xy * cov_xz / cov_yz,
        "y": cov_yy - cov_xy * cov_yz / cov_xz,
        "z": cov_zz - cov_xz * cov_yz / cov_xy,
    }
    signal_variance = {
        "x": cov_xy * cov_xz / cov_yz,
        "y": cov_xy * cov_yz / cov_xz,
        "z": cov_xz * cov_yz / cov_xy,
    }
    snr = {name: signal_variance[name] / var_error[name] for name in "xyz"}
    snr_db = {name: 10.0 * np.log10(snr[name]) for name in "xyz"}
    fractional_mse = {name: 1.0 / (1.0 + snr[name]) for name in "xyz"}
    correlation = {name: 1.0 - fractional_mse[name] for name in "xyz"}

    flags = {
        "condition_corr": (
            (statistics["corrXY"] < corr_th)
            | (statistics["corrXZ"] < corr_th)
            | (statistics["corrYZ"] < corr_th)
        ),
        "condition_n_valid": (
            (~np.isnan(x) & ~np.isnan(y) & ~np.isnan(z)).sum(axis=-1) < nod_th
        ),
        "condition_fMSE": np.logical_or.reduce(
            tuple(
                (fractional_mse[name] < 0.0) | (fractional_mse[name] > 1.0)
                for name in "xyz"
            )
        ),
        "condition_negative_vars_err": np.logical_or.reduce(
            tuple(var_error[name] < 0.0 for name in "xyz")
        ),
    }
    return var_error, snr, snr_db, correlation, fractional_mse, flags


def _calculate_task(task):
    bootstrap_start, bootstrap_stop, i0, i1, j0, j1 = task
    shape = (i1 - i0, j1 - j0, bootstrap_stop - bootstrap_start)
    outputs = [np.full(shape, np.nan, dtype=np.float32) for _ in range(10)]
    chunks = tuple(cube[i0:i1, j0:j1] for cube in _WORKER_CUBES)
    for local_index, bootstrap_index in enumerate(
        range(bootstrap_start, bootstrap_stop)
    ):
        indices = _WORKER_INDICES[bootstrap_index]
        sampled = tuple(np.take(chunk, indices, axis=-1) for chunk in chunks)
        with np.errstate(divide="ignore", invalid="ignore"):
            var_error, snr, _, _, _, flags = _extended_triple_collocation(
                *sampled,
                nod_th=_WORKER_NOD_THRESHOLD,
                corr_th=_WORKER_CORRELATION_THRESHOLD,
            )
        values = (
            var_error["x"],
            var_error["y"],
            var_error["z"],
            snr["x"],
            snr["y"],
            snr["z"],
            flags["condition_corr"],
            flags["condition_n_valid"],
            flags["condition_fMSE"],
            flags["condition_negative_vars_err"],
        )
        for output, value in zip(outputs, values):
            output[..., local_index] = np.asarray(value, dtype=np.float32)
    return bootstrap_start, bootstrap_stop, i0, i1, j0, j1, *outputs


def _tasks(
    latitude_count,
    longitude_count,
    bootstrap_count,
    bootstrap_batch_size,
    chunk_latitude,
    chunk_longitude,
):
    for bootstrap_start in range(0, bootstrap_count, bootstrap_batch_size):
        bootstrap_stop = min(
            bootstrap_start + bootstrap_batch_size, bootstrap_count
        )
        for i0 in range(0, latitude_count, chunk_latitude):
            i1 = min(i0 + chunk_latitude, latitude_count)
            for j0 in range(0, longitude_count, chunk_longitude):
                j1 = min(j0 + chunk_longitude, longitude_count)
                yield bootstrap_start, bootstrap_stop, i0, i1, j0, j1


def _accumulate(result, sums, counts):
    _, _, i0, i1, j0, j1, *arrays = result
    var_errors = arrays[:3]
    snrs = arrays[3:6]
    flags = arrays[6:]
    valid = np.ones(np.shape(flags[0]), dtype=bool)
    for flag in flags:
        valid &= ~np.asarray(flag, dtype=bool)

    metrics = []
    with np.errstate(divide="ignore", invalid="ignore"):
        for value in var_errors:
            value = np.asarray(value, dtype=np.float64)
            metrics.append(
                np.where(valid & (value >= 0.0), np.sqrt(value), np.nan)
            )
        for value in snrs:
            value = np.asarray(value, dtype=np.float64)
            metrics.append(
                np.where(
                    valid & (value > 0.0),
                    np.sqrt(1.0 / (1.0 + 1.0 / value)),
                    np.nan,
                )
            )

    for metric_index, values in enumerate(metrics):
        finite = np.isfinite(values)
        sums[metric_index, i0:i1, j0:j1] += np.where(
            finite, values, 0.0
        ).sum(axis=-1)
        counts[metric_index, i0:i1, j0:j1] += finite.sum(axis=-1).astype(
            np.uint32
        )


def _write_result(
    output_path,
    triplet,
    specs,
    latitude,
    longitude,
    dates,
    sums,
    counts,
    bootstrap_count,
    sample_size,
    random_seed,
    bootstrap_indices_sha256,
    rolling_window,
    nod_threshold,
    correlation_threshold,
    min_valid_bootstrap_fraction,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    if temporary.exists():
        temporary.unlink()
    minimum_count = math.ceil(
        int(bootstrap_count) * float(min_valid_bootstrap_fraction)
    )
    slot_keys = (triplet.ascat, triplet.smap, triplet.reference)
    try:
        with nc.Dataset(temporary, "w", format="NETCDF4") as dataset:
            dataset.createDimension("lat", len(latitude))
            dataset.createDimension("lon", len(longitude))
            lat_variable = dataset.createVariable("lat", "f4", ("lat",))
            lon_variable = dataset.createVariable("lon", "f4", ("lon",))
            lat_variable[:] = latitude
            lon_variable[:] = longitude
            lat_variable.units = "degrees_north"
            lon_variable.units = "degrees_east"
            chunks = (min(20, len(latitude)), min(60, len(longitude)))
            for slot_index, key in enumerate(slot_keys):
                stem = specs[key].stem
                for metric_offset, suffix, long_name in (
                    (0, "err", "bootstrap-mean TCA error standard deviation"),
                    (3, "R", "bootstrap-mean TCA correlation"),
                ):
                    metric_index = metric_offset + slot_index
                    mean = np.full(counts[metric_index].shape, np.nan)
                    np.divide(
                        sums[metric_index],
                        counts[metric_index],
                        out=mean,
                        where=counts[metric_index] > 0,
                    )
                    mean[counts[metric_index] < minimum_count] = np.nan
                    variable = dataset.createVariable(
                        f"{stem}_{suffix}",
                        "f4",
                        ("lat", "lon"),
                        zlib=True,
                        complevel=4,
                        chunksizes=chunks,
                        fill_value=np.float32(np.nan),
                    )
                    variable[:] = mean.astype(np.float32)
                    variable.long_name = long_name
                    if suffix == "err":
                        variable.units = "m3 m-3"
                    else:
                        variable.units = "1"
                    count_variable = dataset.createVariable(
                        f"{stem}_{suffix}_valid_bootstrap_count",
                        "u2",
                        ("lat", "lon"),
                        zlib=True,
                        complevel=4,
                        chunksizes=chunks,
                    )
                    count_variable[:] = counts[metric_index].astype(np.uint16)
            dataset.bootstrap_count = int(bootstrap_count)
            dataset.bootstrap_sample_size = int(sample_size)
            dataset.bootstrap_random_seed = int(random_seed)
            dataset.bootstrap_indices_sha256 = str(bootstrap_indices_sha256)
            dataset.model1 = triplet.ascat.removeprefix("ASCAT_")
            dataset.model2 = triplet.smap.removeprefix("SMAP_")
            dataset.model3 = (
                "ERA5-Land"
                if triplet.reference == "ERA5-Land"
                else triplet.reference
            )
            dataset.rolling_anomaly_window_days = int(rolling_window)
            dataset.minimum_joint_observations = int(nod_threshold)
            dataset.minimum_pairwise_correlation = float(correlation_threshold)
            dataset.minimum_valid_bootstrap_fraction = float(
                min_valid_bootstrap_fraction
            )
            dataset.time_start = str(dates[0])
            dataset.time_end = str(dates[-1])
            dataset.time_steps = int(len(dates))
            dataset.run_complete = 1
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output_path


def _run_triplet(
    triplet,
    specs,
    staged_paths,
    indices_path,
    latitude,
    longitude,
    dates,
    output_directory,
    bootstrap_count,
    sample_size,
    random_seed,
    bootstrap_indices_sha256,
    rolling_window,
    nod_threshold,
    correlation_threshold,
    min_valid_bootstrap_fraction,
    workers,
    bootstrap_batch_size,
    chunk_latitude,
    chunk_longitude,
):
    cube_paths = tuple(
        str(staged_paths[key])
        for key in (triplet.ascat, triplet.smap, triplet.reference)
    )
    sums = np.zeros((6, len(latitude), len(longitude)), dtype=np.float64)
    counts = np.zeros((6, len(latitude), len(longitude)), dtype=np.uint32)
    task_iterator = _tasks(
        len(latitude),
        len(longitude),
        int(bootstrap_count),
        int(bootstrap_batch_size),
        int(chunk_latitude),
        int(chunk_longitude),
    )
    spatial_task_count = math.ceil(len(latitude) / int(chunk_latitude)) * math.ceil(
        len(longitude) / int(chunk_longitude)
    )
    total_tasks = spatial_task_count * math.ceil(
        int(bootstrap_count) / int(bootstrap_batch_size)
    )
    initializer_arguments = (
        cube_paths,
        str(indices_path),
        int(nod_threshold),
        float(correlation_threshold),
    )
    if int(workers) == 1:
        _initialize_worker(*initializer_arguments)
        results = map(_calculate_task, task_iterator)
        for result in tqdm(results, total=total_tasks, desc=triplet.name):
            _accumulate(result, sums, counts)
    else:
        context = mp.get_context("spawn")
        with context.Pool(
            processes=int(workers),
            initializer=_initialize_worker,
            initargs=initializer_arguments,
        ) as pool:
            results = pool.imap_unordered(
                _calculate_task, task_iterator, chunksize=1
            )
            for result in tqdm(results, total=total_tasks, desc=triplet.name):
                _accumulate(result, sums, counts)

    output_path = Path(output_directory) / f"TCA_{triplet.name}.nc"
    return _write_result(
        output_path,
        triplet,
        specs,
        latitude,
        longitude,
        dates,
        sums,
        counts,
        bootstrap_count,
        sample_size,
        random_seed,
        bootstrap_indices_sha256,
        rolling_window,
        nod_threshold,
        correlation_threshold,
        min_valid_bootstrap_fraction,
    )


def run_bootstrap_inventory(
    *,
    triplets,
    product_specs,
    output_directory,
    scratch_directory,
    start_date="2016-03-31",
    end_date="2023-12-31",
    bounds=(-126.0, -66.0, 24.0, 51.0),
    bootstrap_count=5000,
    bootstrap_fraction=0.5,
    random_seed=42,
    rolling_window=101,
    nod_threshold=25,
    correlation_threshold=0.0,
    min_valid_bootstrap_fraction=0.5,
    workers=1,
    bootstrap_batch_size=10,
    chunk_latitude=10,
    chunk_longitude=60,
    anomaly_chunk_latitude=10,
    anomaly_chunk_longitude=60,
):
    """Run fixed-period IID bootstrap TCA and retain only final result grids."""
    triplets = tuple(triplets)
    if not triplets:
        raise ValueError("At least one TCA triplet is required.")
    if int(bootstrap_count) < 1 or not 0.0 < float(bootstrap_fraction) <= 1.0:
        raise ValueError("Bootstrap count/fraction are invalid.")
    if int(rolling_window) <= 1 or int(rolling_window) % 2 == 0:
        raise ValueError("rolling_window must be an odd integer greater than one.")
    if not 0.0 < float(min_valid_bootstrap_fraction) <= 1.0:
        raise ValueError("min_valid_bootstrap_fraction must be in (0, 1].")
    product_specs = {
        key: (
            value
            if isinstance(value, ProductSpec)
            else ProductSpec(**value)
        )
        for key, value in product_specs.items()
    }
    required_keys = {
        key
        for triplet in triplets
        for key in (triplet.ascat, triplet.smap, triplet.reference)
    }
    missing = sorted(required_keys.difference(product_specs))
    if missing:
        raise KeyError(f"Missing product specifications: {missing}")
    dates = np.arange(
        np.datetime64(start_date, "D"),
        np.datetime64(end_date, "D") + np.timedelta64(1, "D"),
        dtype="datetime64[D]",
    )
    if len(dates) <= int(nod_threshold):
        raise ValueError("The comparison period is too short for TCA.")
    sample_size = int(math.floor(len(dates) * float(bootstrap_fraction)))
    if sample_size <= int(nod_threshold):
        raise ValueError("Each bootstrap sample must exceed nod_threshold.")

    first_key = sorted(required_keys)[0]
    first_path = Path(product_specs[first_key].path)
    if not first_path.is_file():
        raise FileNotFoundError(first_path)
    with nc.Dataset(first_path) as first_dataset:
        first_latitude, first_longitude = _coordinate_axes(first_dataset)
        first_latitude_slice, first_longitude_slice = _regional_slice(
            first_latitude, first_longitude, bounds
        )
        target_latitude = first_latitude[first_latitude_slice]
        target_longitude = first_longitude[first_longitude_slice]

    output_directory = Path(output_directory)
    scratch_directory = Path(scratch_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    scratch_directory.mkdir(parents=True, exist_ok=True)
    result_paths = {}
    with tempfile.TemporaryDirectory(
        prefix="tca_", dir=scratch_directory
    ) as temporary_directory:
        temporary_directory = Path(temporary_directory)
        staged_paths = {}
        for key in sorted(required_keys):
            staged_path = temporary_directory / f"{key}_anomaly.npy"
            print(f"Preparing rolling anomalies: {key}")
            staged_paths[key] = _stage_anomaly(
                product_specs[key],
                target_latitude,
                target_longitude,
                dates,
                bounds,
                staged_path,
                int(rolling_window),
                int(anomaly_chunk_latitude),
                int(anomaly_chunk_longitude),
            )
        bootstrap_indices = np.random.default_rng(int(random_seed)).integers(
            0,
            len(dates),
            size=(int(bootstrap_count), sample_size),
            dtype=np.int32,
        )
        bootstrap_indices_sha256 = hashlib.sha256(
            np.ascontiguousarray(bootstrap_indices, dtype=np.int32).tobytes()
        ).hexdigest()
        indices_path = temporary_directory / "bootstrap_indices.npy"
        np.save(indices_path, bootstrap_indices)
        del bootstrap_indices

        for triplet in triplets:
            result_paths[triplet.name] = _run_triplet(
                triplet,
                product_specs,
                staged_paths,
                indices_path,
                target_latitude,
                target_longitude,
                dates,
                output_directory,
                int(bootstrap_count),
                sample_size,
                int(random_seed),
                bootstrap_indices_sha256,
                int(rolling_window),
                int(nod_threshold),
                float(correlation_threshold),
                float(min_valid_bootstrap_fraction),
                int(workers),
                int(bootstrap_batch_size),
                int(chunk_latitude),
                int(chunk_longitude),
            )
    return result_paths
