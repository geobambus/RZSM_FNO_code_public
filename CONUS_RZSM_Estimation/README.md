# CONUS RZSM estimation

This folder builds the static model-input grid, estimates CONUS root-zone soil moisture with the pixel-trained FNO/LSTM ensembles or the exponential filter (EF), merges the yearly estimation files, and calculates held-out triple collocation analysis (TCA). It uses the adjacent `../core/` package and the dependencies in `../requirements.txt`.

## Inputs

Data locations are configured in `core/project_paths.py` (`RZSM_DATA_ROOT`, `RZSM_RESULTS_FP`, `RZSM_FIGURES_FP`). See the [repository README](../README.md) for setup.

The paths below are relative to the configured results root.

| Step | Prepared inputs |
| --- | --- |
| 01 | `Soil_Grids_250m_2.0/Edited_CONUS/Bulk_Density_R_010_CONUS.nc` and `Normalized_Difference_Vegetation_Index/NDVI_averaged.nc` |
| 02–04 | `CONUS_RZSM_Estimation/Model_input.nc` from step 01 and `Soil_Moisture_Active_Passive/SPL3SMP_E.006_2015_2023_R_eqd_010.nc`. Step 02 also reads `Advanced_Scatterometer/ASCAT_2015_2023_conf0_R_eqd_010.nc` when estimating from ASCAT. |
| 05 | Merged estimations from steps 02–04, `ECMWF_Reanalysis_v5_Land/ERA5_Land_2015_2023_R_eqd_010.nc`, and `North_American_Land_Data_Assimilation_System/NLDAS_<model>0125_H_daily_20150401_20231231_R_eqd_010_CONUS_target_7_28cm.nc` for `NOAH`, `VIC`, and `MOS`. |

FNO/LSTM read `Results/Fourier_Neural_Operator/Models/FNO_ensemble.pt` and `Results/Long_Short-Term_Memory/Models/LSTM_ensemble.pt`, written by the pixel workflows (steps 01-04), plus the daily GPM rainfall grid `Results/Global_Precipitation_Measurement/GPM_precipitation_2015_2023_local_06-06.nc` (06:00-06:00 local days).

Step 05 reads merged estimations and prepared ERA5-Land and NLDAS Noah, VIC, and Mosaic references. The EF files must use the configured timescales.

## Run

Set the data root and run from this folder.

```sh
export RZSM_DATA_ROOT=/path/to/folder_containing_Results
python 01_Model_input.py                         # writes Results/CONUS_RZSM_Estimation/Model_input.nc
python 02_EF_RZSM_estimation.py                  # SMAP and ASCAT by default
python 03_LSTM_RZSM_estimation.py                # SMAP
python 04_FNO_RZSM_estimation.py                 # SMAP
```

Run step 05 after producing the merged SMAP estimations and the ASCAT EF reference.

```sh
python 05_TCA_calculation.py                         # all SMAP models and four land references
```

Step 05 runs all 12 triplets by default, with ASCAT EF fixed across the three SMAP estimates and four land references. Each triplet runs bootstrap calculation, flag masking, averaging, and expansion. To inspect the individual stage commands, use `python 05_TCA_calculation.py bootstrap --help` or substitute `mask`, `average`, or `expand`.

## Outputs

- `01_Model_input.py` writes `lat`, `lon`, `CONUS_mask`, `NDVI_avg` and `Bulk_density_RZSM` on the SoilGrids 0.1-degree grid.
- Steps 02–04 each write one NetCDF per year under `Results/CONUS_RZSM_Estimation/Estimations/<model folder>/Yearly/` and automatically combine those yearly files into one NetCDF covering all estimation years for each model and satellite product. The EF recursion is in step 02. The estimators share grid processing and NetCDF merging, and the neural estimators share ensemble inference.
- Each FNO/LSTM output file stores the ten seeded estimations (`RZSM_Trial_1..10`), their mean (`RZSM_avg`) and sample standard deviation (`RZSM_std`).
- EF runs on the CPU with each satellite's T from `core.rzsm_pipeline_config.load_ef_t` (SMAP 10 d, ASCAT 7 d). Outputs go to `Estimations/Exponential_Filter/EF_<product>_estimation.nc` and record `EF_filter_T_days`.

Outputs and per-stage logs go to `Results/CONUS_RZSM_Estimation/Triple_Collocation_Analysis/`. Rerunning a triplet replaces its files. The files record `ASCAT_EF_filter_T_days` and `SMAP_EF_filter_T_days`.

## Options

FNO/LSTM split the ten seeded models across every visible GPU by default (five per card on two cards) and fall back to the CPU when none is visible. Choose cards with `--gpus 0` or `--gpus 0 1 2 3`, or force the CPU with `--gpus cpu`.

TCA uses 40 workers per triplet, bootstrap seed 42, and 5,000 bootstrap replicates by default. Use `--rzsm` or `--model` to select a subset and `--workers` or `--random_seed` to override their defaults.

```sh
python 05_TCA_calculation.py --rzsm FNO --model ERA5 --workers 10
```
