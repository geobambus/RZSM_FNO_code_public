"""Prepare and summarize the separate SMAP-NN comparison experiment."""

import os
from pathlib import Path

import netCDF4 as nc
import numpy as np
from tqdm.auto import tqdm

SMAP_NN_INPUT_START = np.datetime64("2015-04-01", "D")
SMAP_NN_OUTPUT_START = np.datetime64("2016-03-31", "D")
SMAP_NN_END = np.datetime64("2022-03-26", "D")
SMAP_NN_SOURCE_DATASET = (
    "Results of Improved SMAP Soil Moisture Retrieval Using a Deep Neural "
    "Network-Based Replacement of Radiative Transfer and Roughness Model"
)
SMAP_NN_SOURCE_VERSION = "1"
SMAP_NN_SOURCE_DOI = "10.5281/zenodo.13309165"


def _filled(values):
    return np.asarray(np.ma.filled(values, np.nan), dtype=np.float32)


def _axes(dataset):
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
        raise KeyError("Could not find regular latitude/longitude coordinates.")
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


def _time_information(dataset):
    variable = dataset.variables["time"]
    units = getattr(variable, "units", None)
    if not units:
        raise ValueError("SMAP-NN time coordinate has no units.")
    raw = np.asarray(variable[:])
    calendar = getattr(variable, "calendar", "standard")
    decoded = nc.num2date(
        raw,
        units=units,
        calendar=calendar,
        only_use_cftime_datetimes=False,
        only_use_python_datetimes=True,
    )
    dates = np.asarray(
        [
            np.datetime64(
                f"{value.year:04d}-{value.month:02d}-{value.day:02d}", "D"
            )
            for value in decoded
        ],
        dtype="datetime64[D]",
    )
    if len(dates) > 1 and np.any(np.diff(dates) <= np.timedelta64(0, "D")):
        raise ValueError("SMAP-NN dates must be unique and increasing.")
    return raw, units, calendar, dates


def _covering_slice(axis, minimum, maximum, padding=1):
    indices = np.flatnonzero((axis >= minimum) & (axis <= maximum))
    if not len(indices):
        raise ValueError("SMAP-NN grid does not overlap the prediction grid.")
    return slice(
        max(0, int(indices.min()) - int(padding)),
        min(len(axis), int(indices.max()) + int(padding) + 1),
    )


def _write_time_slice(variable, index, values):
    if variable.dimensions == ("lat", "lon", "time"):
        variable[:, :, index] = values
    elif variable.dimensions == ("time", "lat", "lon"):
        variable[index] = values
    else:
        raise ValueError(
            "SMAP-NN source must use (time, lat, lon) or (lat, lon, time)."
        )


def _read_time_slice(variable, index, latitude_slice, longitude_slice):
    if variable.dimensions == ("time", "lat", "lon"):
        return _filled(variable[index, latitude_slice, longitude_slice])
    if variable.dimensions == ("lat", "lon", "time"):
        return _filled(variable[latitude_slice, longitude_slice, index])
    raise ValueError(
        "SMAP-NN source must use (time, lat, lon) or (lat, lon, time)."
    )


def _resample_surface(
    target_longitude,
    target_latitude,
    source_longitude,
    source_latitude,
    values,
    *,
    sampling_method,
    aggregation_method,
    magnification_factor,
):
    """Apply the focused HydroAI resampling behavior needed by SMAP-NN."""
    if (
        target_longitude.shape == source_longitude.shape
        and target_latitude.shape == source_latitude.shape
        and np.array_equal(target_longitude, source_longitude)
        and np.array_equal(target_latitude, source_latitude)
    ):
        return values
    from scipy.interpolate import interp1d
    from scipy.ndimage import zoom

    if int(magnification_factor) > 1:
        factor = int(magnification_factor)
        source_longitude = zoom(source_longitude, factor, order=1)
        source_latitude = zoom(source_latitude, factor, order=1)
        values = zoom(values, factor, order=0)

    valid = (
        np.isfinite(values)
        & (source_latitude <= np.max(target_latitude[:, 0]))
        & (source_latitude > np.min(target_latitude[:, 0]))
        & (source_longitude < np.max(target_longitude[0, :]))
        & (source_longitude >= np.min(target_longitude[0, :]))
    )
    result = np.full(target_latitude.shape, np.nan, dtype=np.float32)
    if not np.any(valid):
        return result
    latitude_to_index = interp1d(
        target_latitude[:, 0],
        np.arange(target_latitude.shape[0]),
        kind=sampling_method,
        bounds_error=False,
    )
    longitude_to_index = interp1d(
        target_longitude[0, :],
        np.arange(target_longitude.shape[1]),
        kind=sampling_method,
        bounds_error=False,
    )
    latitude_index = latitude_to_index(source_latitude[valid])
    longitude_index = longitude_to_index(source_longitude[valid])
    mapped = np.isfinite(latitude_index) & np.isfinite(longitude_index)
    if not np.any(mapped):
        return result
    flat_index = np.ravel_multi_index(
        (
            latitude_index[mapped].astype(int),
            longitude_index[mapped].astype(int),
        ),
        result.shape,
    )
    mapped_values = np.asarray(values[valid][mapped], dtype=np.float64)
    if aggregation_method != "mean":
        raise ValueError(
            "The final SMAP-NN workflow supports aggregation_method='mean'."
        )
    sums = np.bincount(flat_index, weights=mapped_values, minlength=result.size)
    counts = np.bincount(flat_index, minlength=result.size)
    flat_result = result.reshape(-1)
    np.divide(sums, counts, out=flat_result, where=counts > 0)
    return result


def prepare_smap_nn_ef(
    *,
    source_file,
    source_variable,
    target_grid_file,
    output_file,
    time_constant_days=15.0,
    spinup_days=365,
    sampling_method="nearest",
    aggregation_method="mean",
    magnification_factor=4,
):
    """Resample each SMAP-NN map, then update and save its EF RZSM product.

    Resampling is performed before the recursive filter. Only the final EF
    product is retained; the resampled surface-soil-moisture stack is not saved.
    """
    if float(time_constant_days) <= 0:
        raise ValueError("time_constant_days must be positive.")
    if int(magnification_factor) < 1:
        raise ValueError("magnification_factor must be positive.")
    retained_start = SMAP_NN_INPUT_START + np.timedelta64(int(spinup_days), "D")
    if retained_start != SMAP_NN_OUTPUT_START:
        raise ValueError(
            "The final comparison requires the 365-day adjustment ending on "
            f"{SMAP_NN_OUTPUT_START}; received {spinup_days} days."
        )
    source_file = Path(source_file)
    target_grid_file = Path(target_grid_file)
    output_file = Path(output_file)
    if not source_file.is_file():
        raise FileNotFoundError(source_file)
    if not target_grid_file.is_file():
        raise FileNotFoundError(target_grid_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_file.with_name(f".{output_file.name}.{os.getpid()}.tmp")
    if temporary.exists():
        temporary.unlink()

    try:
        with nc.Dataset(source_file) as source, nc.Dataset(
            target_grid_file
        ) as target:
            if source_variable not in source.variables:
                raise KeyError(f"{source_file} has no {source_variable!r} variable.")
            source_data = source.variables[source_variable]
            source_latitude, source_longitude = _axes(source)
            target_latitude, target_longitude = _axes(target)
            raw_time, time_units, calendar, source_dates = _time_information(source)
            expected_dates = np.arange(
                SMAP_NN_INPUT_START,
                SMAP_NN_END + np.timedelta64(1, "D"),
                dtype="datetime64[D]",
            )
            date_lookup = {
                value: index for index, value in enumerate(source_dates)
            }
            missing = [value for value in expected_dates if value not in date_lookup]
            if missing:
                raise ValueError(
                    f"SMAP-NN is missing {len(missing)} required dates; "
                    f"first missing is {missing[0]}."
                )
            time_indices = np.asarray(
                [date_lookup[value] for value in expected_dates], dtype=np.int64
            )
            latitude_slice = _covering_slice(
                source_latitude,
                float(np.min(target_latitude)),
                float(np.max(target_latitude)),
            )
            longitude_slice = _covering_slice(
                source_longitude,
                float(np.min(target_longitude)),
                float(np.max(target_longitude)),
            )
            source_latitude_subset = source_latitude[latitude_slice]
            source_longitude_subset = source_longitude[longitude_slice]
            source_longitude_2d, source_latitude_2d = np.meshgrid(
                source_longitude_subset, source_latitude_subset
            )
            target_longitude_2d, target_latitude_2d = np.meshgrid(
                target_longitude, target_latitude
            )
            elapsed_days = (
                (expected_dates - expected_dates[0]) / np.timedelta64(1, "D")
            ).astype(np.float64)

            with nc.Dataset(temporary, "w", format="NETCDF4") as output:
                output.createDimension("lat", len(target_latitude))
                output.createDimension("lon", len(target_longitude))
                output.createDimension("time", len(expected_dates))
                latitude_variable = output.createVariable(
                    "lat", "f4", ("lat",)
                )
                longitude_variable = output.createVariable(
                    "lon", "f4", ("lon",)
                )
                time_variable = output.createVariable(
                    "time", source.variables["time"].datatype, ("time",)
                )
                latitude_variable[:] = target_latitude
                longitude_variable[:] = target_longitude
                time_variable[:] = raw_time[time_indices]
                latitude_variable.units = "degrees_north"
                longitude_variable.units = "degrees_east"
                time_variable.units = time_units
                time_variable.calendar = calendar
                output.source_dataset = SMAP_NN_SOURCE_DATASET
                output.source_version = SMAP_NN_SOURCE_VERSION
                output.source_doi = SMAP_NN_SOURCE_DOI
                output.source_file = source_file.name
                output.source_variable = source_variable
                output.source_period = "2015-03-31 through 2022-03-26"
                output.retained_input_period = (
                    "2015-04-01 through 2022-03-26"
                )
                output.processing = (
                    "Nearest resampling with mean aggregation to the study "
                    "0.10-degree grid, followed by the study 15-day "
                    "exponential filter and 365-day adjustment"
                )
                rzsm_variable = output.createVariable(
                    "RZSM_prediction",
                    "f4",
                    ("lat", "lon", "time"),
                    zlib=True,
                    complevel=4,
                    chunksizes=(
                        min(10, len(target_latitude)),
                        min(120, len(target_longitude)),
                        1,
                    ),
                    fill_value=np.float32(np.nan),
                )
                rzsm_variable.units = "m3 m-3"
                rzsm_variable.long_name = (
                    "RZSM from exponential filtering of resampled SMAP-NN SSM"
                )

                pixel_count = len(target_latitude) * len(target_longitude)
                current = np.full(pixel_count, np.nan, dtype=np.float32)
                previous_gain = np.ones(pixel_count, dtype=np.float32)
                previous_time = np.zeros(pixel_count, dtype=np.float64)
                initialized = np.zeros(pixel_count, dtype=bool)

                for output_index, source_index in enumerate(
                    tqdm(time_indices, desc="SMAP-NN resampling and EF", unit="day")
                ):
                    surface = _read_time_slice(
                        source_data,
                        int(source_index),
                        latitude_slice,
                        longitude_slice,
                    )
                    surface = np.where(
                        np.isfinite(surface)
                        & (surface > 0.0)
                        & (surface <= 1.0),
                        surface,
                        np.nan,
                    )
                    resampled = _resample_surface(
                        target_longitude_2d,
                        target_latitude_2d,
                        source_longitude_2d,
                        source_latitude_2d,
                        surface,
                        sampling_method=sampling_method,
                        aggregation_method=aggregation_method,
                        magnification_factor=int(magnification_factor),
                    )
                    resampled = np.asarray(resampled, dtype=np.float32)
                    resampled = np.where(
                        np.isfinite(resampled)
                        & (resampled > 0.0)
                        & (resampled <= 1.0),
                        resampled,
                        np.nan,
                    )
                    values = resampled.reshape(-1)
                    valid = np.isfinite(values)
                    new = valid & ~initialized
                    current[new] = values[new]
                    previous_gain[new] = 1.0
                    previous_time[new] = elapsed_days[output_index]
                    initialized[new] = True
                    update = valid & initialized & ~new
                    if np.any(update):
                        time_gap = (
                            elapsed_days[output_index] - previous_time[update]
                        )
                        decay = np.exp(
                            -time_gap / float(time_constant_days)
                        ).astype(np.float32)
                        gain = previous_gain[update] / (
                            previous_gain[update] + decay
                        )
                        current[update] += gain * (
                            values[update] - current[update]
                        )
                        previous_gain[update] = gain
                        previous_time[update] = elapsed_days[output_index]
                    filtered = np.full(pixel_count, np.nan, dtype=np.float32)
                    if elapsed_days[output_index] >= int(spinup_days):
                        filtered[valid] = current[valid]
                    _write_time_slice(
                        rzsm_variable,
                        output_index,
                        filtered.reshape(
                            len(target_latitude), len(target_longitude)
                        ),
                    )
        os.replace(temporary, output_file)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output_file


def _read_summary_grid(path, variable):
    with nc.Dataset(path) as dataset:
        latitude = _filled(dataset.variables["lat"][:])
        longitude = _filled(dataset.variables["lon"][:])
        if variable not in dataset.variables:
            raise KeyError(f"{path} has no {variable!r} variable.")
        values = _filled(dataset.variables[variable][:])
    return latitude, longitude, values


def summarize_smap_nn_comparison(
    *,
    tca_paths,
    output_file,
    fno_stem="RZSM_FNO_SMAP",
    smap_nn_stem="RZSM_SMAP_NN_EF",
):
    """Average the two matched land-reference members and save the comparison."""
    required = (
        "SMAP_FNO_ERA5-Land",
        "SMAP_FNO_NLDAS",
        "SMAP_NN_EF_ERA5-Land",
        "SMAP_NN_EF_NLDAS",
    )
    missing = [key for key in required if key not in tca_paths]
    if missing:
        raise KeyError(f"Missing SMAP-NN TCA results: {missing}")
    bootstrap_settings = set()
    for key in required:
        path = Path(tca_paths[key])
        with nc.Dataset(path) as source:
            bootstrap_settings.add(
                (
                    int(getattr(source, "bootstrap_count", 0)),
                    int(getattr(source, "bootstrap_sample_size", 0)),
                    int(getattr(source, "bootstrap_random_seed", -1)),
                    str(getattr(source, "time_start", "")),
                    str(getattr(source, "time_end", "")),
                )
            )
    if len(bootstrap_settings) != 1:
        raise ValueError(
            "The four SMAP-NN comparison triplets do not share bootstrap settings."
        )
    (
        bootstrap_count,
        bootstrap_sample_size,
        bootstrap_random_seed,
        time_start,
        time_end,
    ) = next(iter(bootstrap_settings))
    if (
        bootstrap_count < 1
        or bootstrap_sample_size < 1
        or bootstrap_random_seed < 0
        or not time_start
        or not time_end
    ):
        raise ValueError("SMAP-NN bootstrap provenance is incomplete.")
    metrics = {}
    latitude = longitude = None
    for product, stem, keys in (
        (
            "SMAP_FNO",
            fno_stem,
            ("SMAP_FNO_ERA5-Land", "SMAP_FNO_NLDAS"),
        ),
        (
            "SMAP_NN_EF",
            smap_nn_stem,
            ("SMAP_NN_EF_ERA5-Land", "SMAP_NN_EF_NLDAS"),
        ),
    ):
        metrics[product] = {}
        for suffix in ("err", "R"):
            members = []
            for key in keys:
                member_latitude, member_longitude, values = _read_summary_grid(
                    tca_paths[key], f"{stem}_{suffix}"
                )
                if latitude is None:
                    latitude, longitude = member_latitude, member_longitude
                elif not (
                    np.array_equal(latitude, member_latitude)
                    and np.array_equal(longitude, member_longitude)
                ):
                    raise ValueError("SMAP-NN TCA result grids are not aligned.")
                members.append(values)
            members = np.stack(members)
            both_valid = np.all(np.isfinite(members), axis=0)
            mean = np.full(members.shape[1:], np.nan, dtype=np.float32)
            mean[both_valid] = members[:, both_valid].mean(axis=0)
            metrics[product][suffix] = mean

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_file.with_name(f".{output_file.name}.{os.getpid()}.tmp")
    if temporary.exists():
        temporary.unlink()
    try:
        with nc.Dataset(temporary, "w", format="NETCDF4") as dataset:
            dataset.createDimension("lat", len(latitude))
            dataset.createDimension("lon", len(longitude))
            latitude_variable = dataset.createVariable("lat", "f4", ("lat",))
            longitude_variable = dataset.createVariable("lon", "f4", ("lon",))
            latitude_variable[:] = latitude
            longitude_variable[:] = longitude
            latitude_variable.units = "degrees_north"
            longitude_variable.units = "degrees_east"
            chunks = (min(20, len(latitude)), min(60, len(longitude)))
            for product in ("SMAP_FNO", "SMAP_NN_EF"):
                for suffix in ("err", "R"):
                    variable = dataset.createVariable(
                        f"{product}_{suffix}",
                        "f4",
                        ("lat", "lon"),
                        zlib=True,
                        complevel=4,
                        chunksizes=chunks,
                        fill_value=np.float32(np.nan),
                    )
                    variable[:] = metrics[product][suffix]
                    variable.units = "m3 m-3" if suffix == "err" else "1"
            difference = dataset.createVariable(
                "SMAP_FNO_minus_SMAP_NN_EF_R",
                "f4",
                ("lat", "lon"),
                zlib=True,
                complevel=4,
                chunksizes=chunks,
                fill_value=np.float32(np.nan),
            )
            difference[:] = metrics["SMAP_FNO"]["R"] - metrics["SMAP_NN_EF"][
                "R"
            ]
            difference.units = "1"
            difference.long_name = "SMAP FNO TCA R minus SMAP NN-EF TCA R"
            dataset.bootstrap_count = bootstrap_count
            dataset.bootstrap_sample_size = bootstrap_sample_size
            dataset.bootstrap_random_seed = bootstrap_random_seed
            dataset.time_start = time_start
            dataset.time_end = time_end
            dataset.source_triplet_count = len(required)
            dataset.target_product_count = 2
            dataset.land_references = "ERA5-Land,NLDAS"
            dataset.R_difference_definition = (
                "SMAP_FNO_R - SMAP_NN_EF_R on pixels finite for both "
                "land-reference means"
            )
        os.replace(temporary, output_file)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output_file
