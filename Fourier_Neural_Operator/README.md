# Pixel-scale FNO

Root-zone soil moisture from SMAP surface soil moisture with a Fourier neural operator over irregular observation events.

Scripts 01–05 contain the stage implementations, including input preparation, Optuna trial execution, candidate refits, ensemble export and pixel inference. `04_FNO_train.py` configures the model and connects it to shared fitting, input and output routines in `../core/`.

## Inputs

Inputs per pixel include the last 32 SMAP observations (SSM, scaled time gap, standardized rainfall since the previous observation) and two static features (NDVI, root-zone bulk density). Ten seeds form the ensemble.

Run from this folder after running [CONUS step 01](../CONUS_RZSM_Estimation/README.md) to prepare the static grid. `R` = `Results/Fourier_Neural_Operator`, `F` = `Figures/Fourier_Neural_Operator` under the data root (see the [repository README](../README.md)); `<area>` is `Train` or `Test`.

Step 01 reads the Train and Test cohort lists and per-pixel NetCDF files with `SMAP_SSM`, `in-situ_RZSM` and daily `Precipitation`.

## Run

### Train with the published settings

Run steps 01, 04, and 05 from this folder. Step 04 uses the published settings in [paper_parameters.py](../core/paper_parameters.py) when `R/Optuna/Best_parameters.json` is absent. If that file exists, step 04 uses the local selection, so use a clean results directory for a run with the published settings.

```
python 01_FNO_input_csv.py
python 04_FNO_train.py
python 05_FNO_RZSM_estimation.py
```

### Run a new hyperparameter search

Run all five steps to search and select hyperparameters before training the final ensemble. Step 02 runs Optuna, and step 03 writes `R/Optuna/Best_parameters.json` for step 04.

```
python 01_FNO_input_csv.py
python 02_FNO_train_optuna.py
python 03_FNO_optuna_trial_candidates.py
python 04_FNO_train.py
python 05_FNO_RZSM_estimation.py
```

The settings were selected using an earlier input set and carried over to the published model. The original FNO search also used an earlier architecture. The release includes the final settings. Historical Optuna databases, trial tables and rankings are kept outside the release.

## Outputs

| Step | Writes |
| --- | --- |
| `01_FNO_input_csv.py` | `R/Inputs/<area>/FNO_input_SMAP.csv`, `R/Inputs/Static_scaler.csv`, `R/Inputs/Rain/<pixel>.npz`, `R/Inputs/Rain/Scaler.csv`, `R/Inputs/Rain/Coverage.csv` |
| `02_FNO_train_optuna.py` (optional) | 200-trial search in `R/Optuna/` |
| `03_FNO_optuna_trial_candidates.py` (optional) | ten-seed review of the 10 best distinct trials, `R/Optuna/Best_parameters.json` |
| `04_FNO_train.py` | `R/Models/FNO_ensemble.pt`, `R/Models/Seed_<seed>/`, loss plots in `F/Models/` |
| `05_FNO_RZSM_estimation.py` | `R/RZSM_estimation/Test/<pixel>_estimation.nc` (ten members, mean, spread) |

Step 05 estimates only the independent test pixels.

## Options

Steps 02–04 accept `--gpu` and `--workers`. Their default GPU indices are `0 1`, and their default worker counts are 20, 40, and 10 respectively. Set `--gpu 0` on a single-GPU machine. Step 05 accepts `--workers` and defaults to 40.
