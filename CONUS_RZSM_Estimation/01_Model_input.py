'''Write the static model predictors (NDVI, bulk density) on the CONUS grid.

Output: ``Results/CONUS_RZSM_Estimation/Model_input.nc`` with ``lat``, ``lon``,
``CONUS_mask``, ``NDVI_avg`` and ``Bulk_density_RZSM``. Daily rainfall for the
event-rainfall models is read separately at estimation time.
'''

import argparse
import hashlib
import json
import os
import sys
from datetime import date, timedelta
from pathlib import Path

PUBLICATION_ROOT = Path(__file__).resolve().parents[1]
if str(PUBLICATION_ROOT) not in sys.path:
    sys.path.insert(0, str(PUBLICATION_ROOT))

import netCDF4 as nc
import numpy as np
from tqdm import tqdm

from core import static_input_profiles as profiles
from core.project_paths import DATA_ROOT, ndvi_average_path, project_results_root

from core.rzsm_pipeline_config import EF_INPUT_START_DATE, VALIDATION_START_DATE


COORD_ATOL = 2e-5
PROCESSING_VERSION = "RZSM_NDVI_fixed_grid_fit_only_v1"
SELECTION_POLICY = "entire_16_day_composite_before_excluded_from_v1"
PRODUCTS = ("MOD13A2", "MYD13A2")


def composite_end(start):
    return min(start + timedelta(days=15), date(start.year, 12, 31))


def coordinates(ds):
    result = []
    for names, axis in ((("lat", "latitude"), 0), (("lon", "longitude"), 1)):
        name = next((name for name in names if name in ds.variables), None)
        if name is None:
            raise ValueError(f"Missing {names} coordinates in {ds.filepath()}")
        var = ds[name]
        if var.ndim == 1:
            values, dimension = var[:], var.dimensions[0]
        elif var.ndim == 2:
            values = var[:, 0] if axis == 0 else var[0, :]
            dimension = var.dimensions[axis]
        else:
            raise ValueError(f"Unsupported coordinate shape: {ds.filepath()}: {name}")
        values = np.asarray(np.ma.filled(values, np.nan), dtype=float)
        diff = np.diff(values)
        if not np.isfinite(values).all() or not (np.all(diff > 0) or np.all(diff < 0)):
            raise ValueError(f"Non-finite/non-monotonic coordinates: {ds.filepath()}: {name}")
        result.append((values, dimension, name))
    return result


def mask_digest(lat, lon, mask):
    digest = hashlib.sha256()
    for values in (np.asarray(lat, dtype="<f4"), np.asarray(lon, dtype="<f4"),
                   np.asarray(mask, dtype="u1")):
        digest.update(str(values.shape).encode("ascii"))
        digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def expected_dates(product, start, cutoff):
    '''MOD13A2 starts on day 1; MYD13A2 is staggered by eight days.'''
    dates = set()
    for year in range(start.year, cutoff.year + 1):
        day = date(year, 1, 1) + timedelta(days=0 if product == "MOD13A2" else 8)
        while day.year == year:
            if day >= start and composite_end(day) < cutoff:
                dates.add(day)
            day += timedelta(days=16)
    return dates


def validate_ndvi_product(ds, expected_lat, expected_lon, expected_start=None):
    '''Reject old full-period means and products made with unrecorded masks.'''
    expected_start = date.fromisoformat(str(EF_INPUT_START_DATE)) if expected_start is None else expected_start
    if (getattr(ds, "ndvi_processing_version", None) != PROCESSING_VERSION
            or getattr(ds, "temporal_selection", None) != SELECTION_POLICY
            or getattr(ds, "run_complete", 0) != 1):
        raise ValueError("NDVI product lacks fixed-grid fitting-period provenance; rerun 02_NDVI_average_std.py.")
    cutoff = date.fromisoformat(ds.excluded_from)
    start = date.fromisoformat(ds.temporal_coverage_start)
    end = date.fromisoformat(ds.temporal_coverage_end)
    if (ds.requested_coverage_start != str(expected_start) or start < expected_start
            or not start <= end < cutoff <= date.fromisoformat(str(VALIDATION_START_DATE))):
        raise ValueError("NDVI must use the study start and exclude whole composites reaching validation/test.")
    wanted = {(product, day) for product in PRODUCTS for day in expected_dates(product, expected_start, cutoff)}
    manifest = json.loads(ds.source_manifest_json)
    actual = [(row["product"], date.fromisoformat(row["date"])) for row in manifest]
    if (ds.source_products != ",".join(PRODUCTS) or ds.number_of_composite_files != len(wanted)
            or len(actual) != len(wanted) or set(actual) != wanted
            or start != min(day for _, day in wanted)
            or end != max(composite_end(day) for _, day in wanted)):
        raise ValueError("NDVI source inventory/dates do not match the complete fitting-period schedule.")
    (lat, _, _), (lon, _, _) = coordinates(ds)
    for actual, expected in ((lat, expected_lat), (lon, expected_lon)):
        if actual.shape != np.shape(expected) or not np.allclose(actual, expected, rtol=0, atol=COORD_ATOL):
            raise ValueError("NDVI product does not match the model grid.")
    mask = np.ma.filled(ds["CONUS_mask"][:], 0)
    if mask.shape != (len(lat), len(lon)) or not np.isin(mask, [0, 1]).all() or not mask.any():
        raise ValueError("NDVI contains an invalid CONUS mask.")
    if ds.grid_mask_sha256 != mask_digest(lat, lon, mask):
        raise ValueError("NDVI coordinate/mask checksum does not match its provenance.")
    count = np.ma.filled(ds["valid_composite_count"][:], 0)
    mean = np.ma.filled(ds["NDVI_averaged"][:], np.nan)
    std = np.ma.filled(ds["NDVI_std"][:], np.nan)
    if any(values.shape != mask.shape for values in (count, mean, std)):
        raise ValueError("NDVI fields have incompatible shapes.")
    valid = mask.astype(bool) & (count > 0)
    if (not valid.any() or np.any(count > ds.number_of_composite_files)
            or not np.array_equal(np.isfinite(mean), valid)
            or not np.array_equal(np.isfinite(std), valid)
            or np.any(count[~mask.astype(bool)] != 0)
            or np.any(std[valid] < 0) or np.any((mean[valid] < -.2001) | (mean[valid] > 1.0001))):
        raise ValueError("NDVI validity/counts/range do not agree with the recorded mask.")
    return mask.astype(bool)


def collapse_rectilinear_coordinate(values, axis, path):
    '''Identify one coordinate axis from a 1-D axis or 2-D mesh grid.'''
    values = np.ma.asarray(values)
    if np.any(np.ma.getmaskarray(values)):
        raise ValueError(f"{path} has masked {axis} coordinates.")
    values = np.asarray(values)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{path} has non-finite {axis} coordinates.")
    if values.ndim == 1:
        return values
    if values.ndim != 2:
        raise ValueError(
            f"{path} has an invalid {axis} coordinate shape: {values.shape}."
        )

    if axis == "latitude":
        coordinate_axis = values[:, 0]
        reconstructed = coordinate_axis[:, None]
    else:
        coordinate_axis = values[0, :]
        reconstructed = coordinate_axis[None, :]
    if not np.allclose(
        values,
        reconstructed,
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError(
            f"{path} uses a curvilinear 2-D {axis} grid; only a "
            "rectilinear latitude/longitude mesh can be aligned safely."
        )
    return coordinate_axis


def read_grid_coordinates(dataset, path):
    '''Read and validate latitude/longitude axes from 1-D or 2-D grids.'''
    coordinate_names = {}
    for axis, candidates in {
        "latitude": ("lat", "latitude"),
        "longitude": ("lon", "longitude"),
    }.items():
        coordinate_names[axis] = next(
            (name for name in candidates if name in dataset.variables),
            None,
        )
        if coordinate_names[axis] is None:
            raise KeyError(
                f"{path} does not contain a recognized {axis} coordinate "
                f"({', '.join(candidates)})."
            )

    latitude = collapse_rectilinear_coordinate(
        dataset.variables[coordinate_names["latitude"]][:],
        "latitude",
        path,
    )
    longitude = collapse_rectilinear_coordinate(
        dataset.variables[coordinate_names["longitude"]][:],
        "longitude",
        path,
    )
    for axis, values in (("latitude", latitude), ("longitude", longitude)):
        if len(values) < 2:
            raise ValueError(
                f"{path} has an invalid {axis} coordinate shape: {values.shape}."
            )
        differences = np.diff(values)
        if not (np.all(differences > 0) or np.all(differences < 0)):
            raise ValueError(
                f"{path} has a non-monotonic {axis} coordinate."
            )
    return latitude, longitude


def validate_grid(dataset, path, expected_lat, expected_lon):
    '''Reject a source whose grid differs from the model-output grid.'''
    latitude, longitude = read_grid_coordinates(dataset, path)
    for axis, actual, expected in (
        ("latitude", latitude, expected_lat),
        ("longitude", longitude, expected_lon),
    ):
        if actual.shape != expected.shape or not np.allclose(
            actual,
            expected,
            rtol=0.0,
            atol=1e-5,
        ):
            raise ValueError(
                f"{path} {axis} grid does not match the model-output grid."
            )


def load_2d_variable(path, variable_name, expected_lat, expected_lon):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with nc.Dataset(path, "r") as ds:
        if variable_name not in ds.variables:
            raise KeyError(f"{path} does not contain {variable_name!r}.")
        validate_grid(ds, path, expected_lat, expected_lon)
        values = np.ma.asarray(ds.variables[variable_name][:]).squeeze()
        return np.ma.array(values, copy=True)


def main(argv=None):
    '''Write the static predictors; daily rainfall is read at estimation time.'''
    argparse.ArgumentParser(description='Prepare the static model predictors on the CONUS grid.').parse_args(argv)
    profile = profiles.EVENT_PROFILE
    soilgrids_results_FP = os.path.join(
        project_results_root(DATA_ROOT),
        "Soil_Grids_250m_2.0",
        "Edited_CONUS",
    )

    bulk_density_FP = os.path.join(soilgrids_results_FP, "Bulk_Density_R_010_CONUS.nc")
    if not os.path.exists(bulk_density_FP):
        raise FileNotFoundError(bulk_density_FP)
    with nc.Dataset(bulk_density_FP, "r") as grid_reference_ds:
        output_lat, output_lon = read_grid_coordinates(
            grid_reference_ds,
            bulk_density_FP,
        )

    NDVI_FP = str(ndvi_average_path())
    with nc.Dataset(NDVI_FP, "r") as ndvi_ds:
        conus_mask = validate_ndvi_product(ndvi_ds, output_lat, output_lon)
        ndvi_source_window = (ndvi_ds.temporal_coverage_start, ndvi_ds.temporal_coverage_end)
        ndvi_excluded_from = ndvi_ds.excluded_from
    NDVI_avg = load_2d_variable(
        NDVI_FP,
        "NDVI_averaged",
        output_lat,
        output_lon,
    )

    bulk_layers = [
        (load_2d_variable(bulk_density_FP, layer_name, output_lat, output_lon) / 100.0, thickness)
        for layer_name, thickness in (("5-15cm", 8.0), ("15-30cm", 13.0))
    ]
    combined_mask = np.logical_or.reduce([
        np.ma.getmaskarray(values) | ~np.isfinite(np.ma.filled(values, np.nan))
        for values, _ in bulk_layers
    ])
    bulk_density_RZSM = np.ma.array(
        sum(np.ma.filled(values, np.nan) * thickness for values, thickness in bulk_layers)
        / sum(thickness for _, thickness in bulk_layers),
        mask=combined_mask,
    )

    data_to_save = {
        "NDVI_avg": NDVI_avg,
        "Bulk_density_RZSM": bulk_density_RZSM,
    }
    expected_variables = profiles.features(profile)
    if list(data_to_save) != expected_variables:
        raise RuntimeError(
            "Model-input variable order does not match the shared static schema: "
            f"expected {expected_variables}, received {list(data_to_save)}."
        )

    variable_metadata = {
        "NDVI_avg": {
            "long_name": "Mean normalized difference vegetation index",
            "units": "1",
        },
        "Bulk_density_RZSM": {
            "long_name": "Soil bulk density, 7-28 cm RZSM target layer",
            "units": "g cm-3",
            "depth_zone": "RZSM",
            "source_layers": "5-15cm,15-30cm",
            "source_weights": "8/21,13/21 overlap-thickness weights",
        },
    }

    output_shape = (len(output_lat), len(output_lon))
    for variable_name, values in data_to_save.items():
        if np.shape(values) != output_shape:
            raise ValueError(
                f"{variable_name} has shape {np.shape(values)}, expected {output_shape}."
            )

    output_FP = str(profiles.grid_path(DATA_ROOT, profile))
    os.makedirs(os.path.dirname(os.path.abspath(output_FP)), exist_ok=True)
    tmp_output_FP = f"{output_FP}.tmp.{os.getpid()}"

    try:
        with nc.Dataset(tmp_output_FP, "w", format="NETCDF4") as output_ds:
            output_ds.static_schema_version = profiles.schema(profile)
            output_ds.run_complete = 0
            output_ds.static_features = ','.join(expected_variables)
            output_ds.static_input_profile = profile
            output_ds.root_zone_target_depth = ">7 cm and <=28 cm"
            output_ds.soilgrids_layer_policy = (
                "RZSM-only static predictors for 7-28 cm from "
                "8/21*(5-15 cm)+13/21*(15-30 cm)"
            )
            output_ds.precipitation_predictors = (
                "dynamic N=1 precipitation is prepared separately; no static precipitation"
            )
            output_ds.product_ssm_climatology_predictors = (
                "excluded to avoid deriving static predictors from the 2015-2023 "
                "dynamic SSM record"
            )
            output_ds.soil_texture_predictors = "no soil texture predictor; bulk density only"
            output_ds.output_grid_source = (
                f"SoilGrids {os.path.basename(bulk_density_FP)} latitude/longitude"
            )
            output_ds.ndvi_source = (
                f"{os.path.basename(NDVI_FP)}; MOD13A2+MYD13A2 composites "
                f"{ndvi_source_window[0]} to {ndvi_source_window[1]}; "
                f"whole composite windows before {ndvi_excluded_from}"
            )
            output_ds.ndvi_excluded_from = ndvi_excluded_from

            output_ds.createDimension("lat", len(output_lat))
            output_ds.createDimension("lon", len(output_lon))

            lat_var = output_ds.createVariable("lat", "f4", ("lat",))
            lon_var = output_ds.createVariable("lon", "f4", ("lon",))
            lat_var.units = "degrees_north"
            lon_var.units = "degrees_east"
            lat_var[:] = output_lat
            lon_var[:] = output_lon

            conus_var = output_ds.createVariable(
                "CONUS_mask",
                "u1",
                ("lat", "lon"),
                zlib=True,
            )
            conus_var[:] = conus_mask.astype(np.uint8)
            conus_var.long_name = "Contiguous United States land mask"
            conus_var.flag_values = np.asarray([0, 1], dtype=np.uint8)
            conus_var.flag_meanings = "outside_CONUS inside_CONUS"
            conus_var.source = (
                "regionmask Natural Earth v5.0.0 countries_110; "
                "United States region 4; 24-51N, 126-66W"
            )

            for variable_name, values in tqdm(
                data_to_save.items(),
                desc="Saving model predictors",
                leave=False,
            ):
                variable = output_ds.createVariable(
                    variable_name,
                    "f4",
                    ("lat", "lon"),
                    zlib=True,
                    fill_value=np.float32(np.nan),
                )
                masked_values = np.where(
                    conus_mask,
                    np.ma.filled(values, np.nan),
                    np.nan,
                )
                variable[:] = masked_values.astype(np.float32)
                for attr_name, attr_value in variable_metadata[variable_name].items():
                    variable.setncattr(attr_name, attr_value)
            output_ds.run_complete = 1

        os.replace(tmp_output_FP, output_FP)
    except Exception:
        if os.path.exists(tmp_output_FP):
            os.remove(tmp_output_FP)
        raise


if __name__ == '__main__':
    main()
    print("01_Model_input.py is finished.")
