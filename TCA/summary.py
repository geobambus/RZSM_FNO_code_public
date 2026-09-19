"""Compact product-level summaries of the final fixed-EF TCA inventory."""

import os
from pathlib import Path

import netCDF4 as nc
import numpy as np


OUTPUT_NAMES = {
    "ASCAT_EF": "ASCAT_EF_R",
    "ASCAT_FNO": "ASCAT_FNO_R",
    "ASCAT_LSTM": "ASCAT_LSTM_R",
    "SMAP_EF": "SMAP_EF_R",
    "SMAP_FNO": "SMAP_FNO_R",
    "SMAP_LSTM": "SMAP_LSTM_R",
    "ERA5-Land": "ERA5-Land_R",
    "NLDAS_NOAH": "NLDAS_NOAH_R",
    "NLDAS_VIC": "NLDAS_VIC_R",
    "NLDAS_MOSAIC": "NLDAS_MOSAIC_R",
}


def _filled(values):
    return np.asarray(np.ma.filled(values, np.nan), dtype=np.float32)


def _selected_products(triplet):
    selected = []
    if triplet.smap == "SMAP_EF":
        selected.append(triplet.ascat)
    if triplet.ascat == "ASCAT_EF":
        selected.append(triplet.smap)
    selected.append(triplet.reference)
    return tuple(selected)


def summarize_fixed_inventory(
    *,
    tca_paths,
    triplets,
    product_specs,
    output_file,
):
    """Average each target across its four matched land-reference triplets."""
    triplets = tuple(triplets)
    sums = {}
    counts = {}
    expected_counts = {key: 0 for key in OUTPUT_NAMES}
    latitude = longitude = None

    for triplet in triplets:
        if triplet.name not in tca_paths:
            raise KeyError(f"Missing TCA result for {triplet.name}.")
        path = Path(tca_paths[triplet.name])
        with nc.Dataset(path) as dataset:
            member_latitude = _filled(dataset.variables["lat"][:])
            member_longitude = _filled(dataset.variables["lon"][:])
            if latitude is None:
                latitude, longitude = member_latitude, member_longitude
            elif not (
                np.array_equal(latitude, member_latitude)
                and np.array_equal(longitude, member_longitude)
            ):
                raise ValueError("TCA result grids are not aligned.")
            for key in _selected_products(triplet):
                expected_counts[key] += 1
                variable_name = f"{product_specs[key].stem}_R"
                if variable_name not in dataset.variables:
                    raise KeyError(f"{path} has no {variable_name!r} variable.")
                values = _filled(dataset.variables[variable_name][:])
                if key not in sums:
                    sums[key] = np.zeros(values.shape, dtype=np.float64)
                    counts[key] = np.zeros(values.shape, dtype=np.uint8)
                valid = np.isfinite(values)
                sums[key][valid] += values[valid]
                counts[key] += valid.astype(np.uint8)

    missing_products = [key for key in OUTPUT_NAMES if key not in sums]
    if missing_products:
        raise RuntimeError(
            f"The fixed triplet inventory did not score: {missing_products}"
        )
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
            for key, output_name in OUTPUT_NAMES.items():
                mean = np.full(counts[key].shape, np.nan, dtype=np.float32)
                np.divide(
                    sums[key], counts[key], out=mean, where=counts[key] > 0
                )
                variable = dataset.createVariable(
                    output_name,
                    "f4",
                    ("lat", "lon"),
                    zlib=True,
                    complevel=4,
                    chunksizes=chunks,
                    fill_value=np.float32(np.nan),
                )
                variable[:] = mean
                variable.units = "1"
                variable.long_name = "Mean TCA correlation across matched triplets"
                count_variable = dataset.createVariable(
                    f"{output_name}_valid_triplet_count",
                    "u1",
                    ("lat", "lon"),
                    zlib=True,
                    complevel=4,
                    chunksizes=chunks,
                )
                count_variable[:] = counts[key]
                count_variable.expected_triplet_count = expected_counts[key]
            dataset.source_triplet_count = len(triplets)
        os.replace(temporary, output_file)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output_file
