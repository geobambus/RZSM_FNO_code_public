# RZSM FNO prediction

Python workflows for exponential-filter (EF), long short-term memory (LSTM),
and Fourier neural operator (FNO) root-zone soil-moisture prediction at ISMN
pixels and across the contiguous United States (CONUS), followed by bootstrap
triple-collocation analysis (TCA).

## Scope

This repository contains the prediction, model-training, and TCA code used by
the project. It does **not** distribute or preprocess the source ISMN, ASCAT,
SMAP, ERA5-Land, or NLDAS datasets. Users must obtain those datasets under
their respective licenses and prepare the input files described below.

## Installation

Create the Conda environment and start JupyterLab from the repository root:

```bash
conda env create -f environment.yml
conda activate rzsm-fno
jupyter lab
```

Python 3.12 is the documented runtime. CUDA is optional; the notebooks use
PyTorch CPU execution when CUDA is unavailable.

## Configuration

By default, user-supplied inputs and generated products are expected under
`Data/` and `Results/` in the repository. Paths can be changed without editing
the notebooks:

```bash
export RZSM_DATA_ROOT=/path/to/input/data
export RZSM_RESULTS_FP=/path/to/results
export RZSM_FIGURES_FP=/path/to/figures
```

Worker counts can also be set through `EF_WORKERS`,
`LSTM_TRAINING_WORKERS`, `LSTM_PREDICTION_WORKERS`,
`FNO_TRAINING_WORKERS`, `FNO_PREDICTION_WORKERS`, and `TCA_WORKERS`.

## Required prepared inputs

The notebooks expect the following products. These files are intentionally not
stored in Git because they are derived from externally distributed scientific
datasets and can be large.

```text
Results/
├── ISMN/
│   ├── Station_TCA_Screening/
│   │   ├── Train_pixel_list.csv
│   │   ├── Test_pixel_list.csv
│   │   └── Excluded_pixel_list.csv
│   ├── ISMN_Train/{ASCAT,SMAP}/<lat_idx>_<lon_idx>.nc
│   ├── ISMN_Test/{ASCAT,SMAP}/<lat_idx>_<lon_idx>.nc
│   └── ISMN_Excluded/{ASCAT,SMAP}/<lat_idx>_<lon_idx>.nc
├── CONUS_Prediction/Model_input_static_eqd_010.nc
├── ASCAT/ASCAT_20150401_20231231_eqd_010.nc
├── SMAP/SMAP_20150401_20231231_eqd_010.nc
├── ERA5-Land/ERA5_Land_20150401_20231231_eqd_010.nc
└── NLDAS/
    ├── NLDAS_NOAH_20150401_20231231_eqd_010.nc
    ├── NLDAS_VIC_20150401_20231231_eqd_010.nc
    └── NLDAS_MOSAIC_20150401_20231231_eqd_010.nc
```

Point files must cover the 3,197 daily timestamps from 2015-04-01 through
2023-12-31. They must contain `time`, `in-situ_RZSM`, and the appropriate
satellite surface-soil-moisture variable (`ASCAT_SSM` or `SMAP_SSM`). Users do
not need to prepare an EF variable: both EF notebooks calculate predictions
directly from SSM.

The static CONUS file must contain `CONUS_mask` and the seven predictors named
in `LSTM/settings.py`. ASCAT and SMAP CONUS files must use the matching 0.1°
grid and daily time coordinate. TCA additionally requires the ERA5-Land and
three NLDAS land-surface-model files shown above.

Pretrained LSTM and FNO ensemble bundles are required for prediction-only
runs. They can either be generated with the training notebooks or placed at:

```text
Results/LSTM/Train/LSTM_{ASCAT,SMAP}_ensemble.pt
Results/FNO/Train/FNO_{ASCAT,SMAP}_ensemble.pt
```

Point-scale EF products follow this layout:

```text
Results/EF/
├── Fixed_EF/{Train,Test,Excluded}/{ASCAT,SMAP}/<lat_idx>_<lon_idx>_prediction.nc
└── Calibrated_EF/
    ├── Calibration/
    │   ├── Calibrated_EF_candidate_scores.csv
    │   ├── Calibrated_EF_pixel_optimal_T.csv
    │   └── Calibrated_EF_T_summary.json
    └── Test/{ASCAT,SMAP}/<lat_idx>_<lon_idx>_prediction.nc
```

## Workflow order

1. Place the prepared input files in the structure above, or configure their
   parent directory through the environment variables.
2. Run `LSTM_train.ipynb` and `FNO_train.ipynb` if pretrained bundles are not
   already available.
3. Run `Fixed_EF_prediction.ipynb`, `LSTM_prediction.ipynb`, and
   `FNO_prediction.ipynb` for the primary point-scale products. Run
   `Calibrated_EF_prediction.ipynb` for the development-calibrated point
   sensitivity.
4. Run `CONUS_RZSM_prediction.ipynb` to generate the six CONUS Fixed EF/LSTM/FNO products
   for ASCAT and SMAP.
5. Run `TCA_calculation.ipynb` after the six predictions and the four land-model
   reference files are available.

Each notebook performs explicit prerequisite checks and documents its
scientific and runtime settings near the top. Training the ten-member neural
ensembles and running the CONUS/TCA calculations can require substantial CPU
or GPU time, memory, and storage.

## Repository layout

- `EF/`: shared Fixed EF and Calibrated EF filtering, calibration, and
  point-prediction helpers.
- `LSTM/`: LSTM data, model, training, and prediction implementation.
- `FNO/`: FNO data, model, training, and prediction implementation.
- `CONUS_RZSM_Prediction/`: CONUS static-input and prediction helpers.
- `TCA/`: bootstrap TCA calculation and summary helpers.

## Outputs

Numerical outputs are written below `Results/`. Figures from the EF evaluation
are written below `Outputs/Figures/`. Both roots can be redirected through
`config.py` environment variables.

## Data availability

Users are responsible for obtaining ISMN, ASCAT, SMAP, ERA5-Land, and NLDAS
data and complying with the providers' terms. Dataset-specific download links,
versions, preprocessing definitions, and citations should be added here before
the archival public release.

## Citation and license

Project citation and software-license information will be added before the
archival public release.
