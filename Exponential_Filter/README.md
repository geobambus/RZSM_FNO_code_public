# Pixel-scale exponential filter (EF)

Root-zone soil moisture from SMAP surface soil moisture with the observation-time exponential filter defined in step 01 and called by step 02. The filter state updates only on valid SSM observations (SSM outside (0, 1] is rejected), weighted by the elapsed time and T; days without an observation stay NaN, and the first 365 days of the record are masked as spin-up.

Every EF product, pixel or gridded, uses one T per satellite from `core.rzsm_pipeline_config.load_ef_t` (SMAP 10 d, ASCAT 7 d). Step 01 shows how these values were selected (supplementary Fig. S1).

## Inputs

The paths below use `R` for `Results/Exponential_Filter` and `I` for `Results/International_Soil_Moisture_Network` under the configured data root. See the [repository README](../README.md) for setup.

The pixel files `<lat_idx>_<lon_idx>.nc` (not part of this repository) hold `time` and `in-situ_RZSM`. Step 01 reads `SMAP_SSM` and `ASCAT_SSM` from the training files. Step 02 reads `SMAP_SSM` from the Test files. Gridded SMAP and ASCAT estimations are generated in `../CONUS_RZSM_Estimation/`.

## Run

Run from this folder.

```sh
python 01_EF_T_calibration.py
python 02_EF_RZSM_estimation.py
```

Step 01 uses fixed study calibration settings. To generate Test estimations, step 02 can run on its own with the prepared Test inputs. It reads the configured SMAP T of 10 days directly.

**01 Calibration** follows Pasik et al. (2023, GMD 16:4957, Sect. 3.1.2). It tries integer T from 1 to 100 d and finds the T that maximizes each SCAN training pixel's Pearson R on 2016-03-31 through 2022-06-12 (at least 100 pairs). Pixels whose best-T R is below 0.5 or not significant (two-sided p >= 0.05, Fisher z) are left out, and the global T is the median optimum of the remaining pixels, snapped to the candidate grid. Ties favour T closest to 15 d, then the smaller T. Test-period targets are never read. As in the gridded EF, 0 is a valid ASCAT value. `EF_T_calibration_pixel_optima.csv` keeps every pixel with its `p_value` and `used_for_selection` flag.

## Outputs

| Step | Reads | Writes |
| --- | --- | --- |
| `01_EF_T_calibration.py` | `I/ISMN_Train/<pixel>.nc`, `I/Station_TCA_Screening/Common_Train_pixel_list.csv` | `R/T_calibration/EF_T_calibration_{candidate_scores.csv,pixel_optima.csv,summary.json}` |
| `02_EF_RZSM_estimation.py` | `I/ISMN_Test/<pixel>.nc`, `I/Station_TCA_Screening/Common_Test_pixel_list.csv` | `R/RZSM_estimation/Test/<pixel>_estimation.nc` |

**02 Estimation files** hold `SSM`, observed `RZSM` and `RZSM_estimation` on `time`, with `filter_T_days`, `filter_policy`, `spin_up_days` and `pipeline_version` attributes. Every listed pixel gets a file, all NaN when it has no valid SMAP data.

## Options

Use `--workers` to change the default of 60 processes.
