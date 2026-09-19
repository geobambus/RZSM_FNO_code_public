"""Build the seven-layer static input used by point and CONUS models."""

from __future__ import annotations

import os
from pathlib import Path

import netCDF4 as nc
import numpy as np

from LSTM.settings import STATIC_FEATURES


COORDINATE_NAMES = {
    "lat": ("lat", "latitude", "y"),
    "lon": ("lon", "longitude", "x"),
}


def _filled(values, dtype=np.float32):
    return np.asarray(np.ma.filled(values, np.nan), dtype=dtype)


def _coordinate_name(dataset, axis):
    for name in COORDINATE_NAMES[axis]:
        if name in dataset.variables:
            return name
    raise KeyError(f"{dataset.filepath()} has no {axis} coordinate.")


def _coordinate_axis(dataset, axis):
    name = _coordinate_name(dataset, axis)
    values = _filled(dataset.variables[name][:], dtype=np.float64)
    if values.ndim == 1:
        return name, values
    if values.ndim != 2:
        raise ValueError(f"{name!r} must be a one- or two-dimensional coordinate.")
    first_column = values[:, 0]
    first_row = values[0, :]
    if axis == "lat":
        if np.allclose(values, first_column[:, None], atol=1e-6, equal_nan=True):
            return name, first_column
        if np.allclose(values, first_row[None, :], atol=1e-6, equal_nan=True):
            return name, first_row
    else:
        if np.allclose(values, first_row[None, :], atol=1e-6, equal_nan=True):
            return name, first_row
        if np.allclose(values, first_column[:, None], atol=1e-6, equal_nan=True):
            return name, first_column
    raise ValueError(f"{name!r} is not a regular latitude/longitude coordinate.")


def _coordinate_indices(source, target, label, tolerance=0.051):
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.ndim != 1 or target.ndim != 1:
        raise ValueError(f"{label} coordinates must be one-dimensional.")
    if len(source) == len(target) and np.allclose(
        source, target, atol=tolerance, rtol=0.0
    ):
        return np.arange(len(target), dtype=np.int64)
    if len(source) == len(target) and np.allclose(
        source[::-1], target, atol=tolerance, rtol=0.0
    ):
        return np.arange(len(source) - 1, -1, -1, dtype=np.int64)

    order = np.argsort(source)
    sorted_source = source[order]
    insertion = np.searchsorted(sorted_source, target)
    upper = np.clip(insertion, 0, len(source) - 1)
    lower = np.clip(insertion - 1, 0, len(source) - 1)
    choose_lower = np.abs(sorted_source[lower] - target) <= np.abs(
        sorted_source[upper] - target
    )
    selected = np.where(choose_lower, lower, upper)
    indices = order[selected]
    differences = np.abs(source[indices] - target)
    if np.any(differences > tolerance):
        raise ValueError(
            f"{label} cannot be aligned to the model grid; maximum difference "
            f"is {float(np.nanmax(differences)):.6g} degrees."
        )
    if len(np.unique(indices)) != len(indices):
        raise ValueError(f"{label} alignment reuses source coordinates.")
    return indices.astype(np.int64)


def _spatial_dimensions(variable, latitude_name, longitude_name):
    dimensions = list(variable.dimensions)
    latitude_dimension = latitude_name if latitude_name in dimensions else None
    longitude_dimension = longitude_name if longitude_name in dimensions else None
    if latitude_dimension is None:
        latitude_dimension = next(
            (name for name in dimensions if name.lower() in {"lat", "latitude", "y"}),
            None,
        )
    if longitude_dimension is None:
        longitude_dimension = next(
            (name for name in dimensions if name.lower() in {"lon", "longitude", "x"}),
            None,
        )
    if latitude_dimension is None or longitude_dimension is None:
        raise ValueError(f"Could not identify spatial dimensions for {variable.name!r}.")
    return latitude_dimension, longitude_dimension


def read_aligned_grid(path, variable_name, target_latitude, target_longitude):
    """Read one two-dimensional field and align it to the target axes."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with nc.Dataset(path) as dataset:
        if variable_name not in dataset.variables:
            raise KeyError(f"{path} has no {variable_name!r} variable.")
        latitude_name, source_latitude = _coordinate_axis(dataset, "lat")
        longitude_name, source_longitude = _coordinate_axis(dataset, "lon")
        variable = dataset.variables[variable_name]
        latitude_dimension, longitude_dimension = _spatial_dimensions(
            variable, latitude_name, longitude_name
        )
        selections = []
        retained_dimensions = []
        for dimension in variable.dimensions:
            if dimension == latitude_dimension:
                selections.append(slice(None))
                retained_dimensions.append("lat")
            elif dimension == longitude_dimension:
                selections.append(slice(None))
                retained_dimensions.append("lon")
            else:
                if len(dataset.dimensions[dimension]) != 1:
                    raise ValueError(
                        f"{path}:{variable_name} has unsupported non-spatial "
                        f"dimension {dimension!r}."
                    )
                selections.append(0)
        values = _filled(variable[tuple(selections)])
        if retained_dimensions == ["lon", "lat"]:
            values = values.T
        elif retained_dimensions != ["lat", "lon"]:
            raise ValueError(
                f"Unexpected retained dimensions for {path}:{variable_name}: "
                f"{retained_dimensions}."
            )
        try:
            latitude_indices = _coordinate_indices(
                source_latitude, target_latitude, f"{path.name} latitude"
            )
            longitude_indices = _coordinate_indices(
                source_longitude, target_longitude, f"{path.name} longitude"
            )
        except ValueError:
            # Regional products such as the NASADEM mask cover only CONUS.
            # Place their coordinates on the complete global target axes.
            target_latitude_indices = _coordinate_indices(
                target_latitude, source_latitude, f"{path.name} latitude"
            )
            target_longitude_indices = _coordinate_indices(
                target_longitude, source_longitude, f"{path.name} longitude"
            )
            output = np.full(
                (len(target_latitude), len(target_longitude)),
                np.nan,
                dtype=np.float32,
            )
            output[np.ix_(target_latitude_indices, target_longitude_indices)] = values
            return output
        return values[np.ix_(latitude_indices, longitude_indices)]


def depth_weighted_composite(shallow, deep, shallow_thickness=8.0, deep_thickness=13.0):
    """Combine SoilGrids 5--15 and 15--30 cm over the 7--28 cm target."""
    shallow = _filled(shallow)
    deep = _filled(deep)
    valid = np.isfinite(shallow) & np.isfinite(deep)
    output = np.full(shallow.shape, np.nan, dtype=np.float32)
    denominator = float(shallow_thickness) + float(deep_thickness)
    output[valid] = (
        shallow[valid] * float(shallow_thickness)
        + deep[valid] * float(deep_thickness)
    ) / denominator
    return output


def _reference_axes(
    reference_file,
    target_resolution,
    require_complete_global_axes,
):
    reference_file = Path(reference_file)
    if not reference_file.is_file():
        raise FileNotFoundError(reference_file)
    with nc.Dataset(reference_file) as dataset:
        _, latitude = _coordinate_axis(dataset, "lat")
        _, longitude = _coordinate_axis(dataset, "lon")
    resolution = float(target_resolution)
    for label, axis in (("latitude", latitude), ("longitude", longitude)):
        differences = np.abs(np.diff(axis))
        if not np.allclose(differences, resolution, atol=1e-4, rtol=0.0):
            raise ValueError(
                f"{reference_file} does not have a regular {resolution:.2f}-degree "
                f"{label} axis."
            )
    if require_complete_global_axes:
        expected_latitude_count = int(round(180.0 / resolution))
        expected_longitude_count = int(round(360.0 / resolution))
        if (
            len(latitude) != expected_latitude_count
            or len(longitude) != expected_longitude_count
        ):
            raise ValueError(
                "The model-input reference must retain complete global axes; "
                f"found {len(latitude)} x {len(longitude)} instead of "
                f"{expected_latitude_count} x {expected_longitude_count}."
            )
    return latitude.astype(np.float32), longitude.astype(np.float32)


def create_model_input(
    *,
    reference_file,
    lst_file,
    ndvi_file,
    soil_files,
    conus_mask_file,
    output_file,
    lst_variable="LST_avg",
    ndvi_variable="NDVI_averaged",
    mask_variable="CONUS_mask",
    shallow_layer="5-15cm",
    deep_layer="15-30cm",
    shallow_thickness_cm=8.0,
    deep_thickness_cm=13.0,
    soil_scales=None,
    target_resolution=0.10,
    require_complete_global_axes=True,
):
    """Create the full-grid, CONUS-masked seven-feature model input.

    Soil values represent 7--28 cm: 8/21 of the 5--15 cm layer and 13/21
    of the 15--30 cm layer. Both source layers must be finite. The full global
    0.10-degree coordinate axes are retained, while values outside CONUS are
    stored as NaN.
    """
    soil_scales = soil_scales or {
        "Bulk_density_RZSM": 100.0,
        "Sand_RZSM": 1000.0,
        "Clay_RZSM": 1000.0,
        "Water_content_10kPa_RZSM": 1000.0,
        "Water_content_1500kPa_RZSM": 1000.0,
    }
    if set(soil_files) != set(soil_scales):
        raise ValueError("soil_files and soil_scales must contain the same features.")
    expected = ("LST_avg", "NDVI_avg", *soil_files.keys())
    if tuple(expected) != tuple(STATIC_FEATURES):
        raise ValueError(
            f"Model-input feature order must be {STATIC_FEATURES}; received {expected}."
        )

    latitude, longitude = _reference_axes(
        reference_file,
        target_resolution,
        require_complete_global_axes,
    )
    conus_mask = read_aligned_grid(
        conus_mask_file, mask_variable, latitude, longitude
    )
    conus_mask = np.isfinite(conus_mask) & (conus_mask > 0)
    if not np.any(conus_mask):
        raise ValueError("The aligned CONUS mask contains no retained cells.")

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_file.with_name(f".{output_file.name}.{os.getpid()}.tmp")
    if temporary.exists():
        temporary.unlink()

    metadata = {
        "LST_avg": ("Mean MODIS land-surface temperature", "K"),
        "NDVI_avg": ("Mean normalized difference vegetation index", "1"),
        "Bulk_density_RZSM": ("Soil bulk density, 7-28 cm", "g cm-3"),
        "Sand_RZSM": ("Soil sand fraction, 7-28 cm", "1"),
        "Clay_RZSM": ("Soil clay fraction, 7-28 cm", "1"),
        "Water_content_10kPa_RZSM": ("Soil water content at 10 kPa, 7-28 cm", "m3 m-3"),
        "Water_content_1500kPa_RZSM": ("Soil water content at 1500 kPa, 7-28 cm", "m3 m-3"),
    }
    try:
        with nc.Dataset(temporary, "w", format="NETCDF4") as output:
            output.createDimension("lat", len(latitude))
            output.createDimension("lon", len(longitude))
            lat_variable = output.createVariable("lat", "f4", ("lat",))
            lon_variable = output.createVariable("lon", "f4", ("lon",))
            lat_variable[:] = latitude
            lon_variable[:] = longitude
            lat_variable.units = "degrees_north"
            lon_variable.units = "degrees_east"
            mask_output = output.createVariable(
                "CONUS_mask", "u1", ("lat", "lon"), zlib=True, complevel=4
            )
            mask_output[:] = conus_mask.astype(np.uint8)
            mask_output.flag_values = np.asarray([0, 1], dtype=np.uint8)
            mask_output.flag_meanings = "outside_CONUS inside_CONUS"
            mask_output.source = str(conus_mask_file)

            feature_grids = {
                "LST_avg": read_aligned_grid(
                    lst_file, lst_variable, latitude, longitude
                ),
                "NDVI_avg": read_aligned_grid(
                    ndvi_file, ndvi_variable, latitude, longitude
                ),
            }
            for feature_name, source_file in soil_files.items():
                scale = float(soil_scales[feature_name])
                shallow = read_aligned_grid(
                    source_file, shallow_layer, latitude, longitude
                ) / scale
                deep = read_aligned_grid(
                    source_file, deep_layer, latitude, longitude
                ) / scale
                feature_grids[feature_name] = depth_weighted_composite(
                    shallow,
                    deep,
                    shallow_thickness=shallow_thickness_cm,
                    deep_thickness=deep_thickness_cm,
                )

            for feature_name in STATIC_FEATURES:
                values = np.asarray(feature_grids.pop(feature_name), dtype=np.float32)
                values[~conus_mask] = np.nan
                variable = output.createVariable(
                    feature_name,
                    "f4",
                    ("lat", "lon"),
                    zlib=True,
                    complevel=4,
                    chunksizes=(min(300, len(latitude)), min(600, len(longitude))),
                    fill_value=np.float32(np.nan),
                )
                variable[:] = values
                variable.long_name, variable.units = metadata[feature_name]
                if feature_name.endswith("_RZSM"):
                    variable.depth_zone = "7-28 cm"
                    variable.source_layers = f"{shallow_layer},{deep_layer}"
                    variable.source_weights = (
                        f"{shallow_thickness_cm}/"
                        f"{shallow_thickness_cm + deep_thickness_cm},"
                        f"{deep_thickness_cm}/"
                        f"{shallow_thickness_cm + deep_thickness_cm}"
                    )

            output.title = "Static inputs for RZSM prediction"
            output.feature_order = ",".join(STATIC_FEATURES)
            output.root_zone_target_depth = "7 < depth <= 28 cm"
            output.target_resolution_degrees = float(target_resolution)
            output.grid_policy = (
                "Complete global 0.10-degree coordinate axes with finite values "
                "retained only inside the CONUS mask"
            )
            output.soil_layer_policy = (
                "8/21*(5-15 cm) + 13/21*(15-30 cm); both layers required"
            )
        os.replace(temporary, output_file)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output_file


def inspect_model_input(path):
    """Return a compact validation summary for a completed model input."""
    path = Path(path)
    with nc.Dataset(path) as dataset:
        missing = [name for name in ("lat", "lon", "CONUS_mask", *STATIC_FEATURES) if name not in dataset.variables]
        if missing:
            raise KeyError(f"{path} is missing variables: {missing}")
        mask = np.asarray(dataset.variables["CONUS_mask"][:], dtype=bool)
        rows = []
        for name in STATIC_FEATURES:
            values = _filled(dataset.variables[name][:])
            rows.append(
                {
                    "variable": name,
                    "shape": values.shape,
                    "finite_CONUS_pixels": int(np.count_nonzero(np.isfinite(values) & mask)),
                    "outside_mask_finite_pixels": int(np.count_nonzero(np.isfinite(values) & ~mask)),
                }
            )
    return rows
