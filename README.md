# Root-zone soil moisture estimation and triple collocation

Python workflows for pixel-scale root-zone soil moisture estimation from SMAP, CONUS estimation, and held-out triple collocation analysis (TCA). The pixel models are an exponential filter (EF), a Fourier neural operator (FNO), and a long short-term memory network (LSTM).

| Folder | Workflow |
| --- | --- |
| [Exponential_Filter](Exponential_Filter/README.md) | Calibrate the SMAP and ASCAT EF timescales and estimate from SMAP at Test pixels. |
| [Fourier_Neural_Operator](Fourier_Neural_Operator/README.md) | Prepare pixel inputs, train or tune the FNO, and estimate at Test pixels. |
| [Long_Short-Term_Memory](Long_Short-Term_Memory/README.md) | Prepare pixel inputs, train or tune the LSTM, and estimate at Test pixels. |
| [CONUS_RZSM_Estimation](CONUS_RZSM_Estimation/README.md) | Prepare the static grid, estimate and merge CONUS RZSM, and run TCA. |
| [core](core/README.md) | Shared paths, study settings, model architectures, and pixel training routines. |

## Setup

Run the commands from the repository root. The study workflows ran on Linux with Python 3.12.11. Create a virtual environment and install the pinned dependencies from [requirements.txt](requirements.txt). For GPU execution, use a PyTorch build compatible with the system's CUDA setup. Optuna is used by neural steps 02 and 03 for hyperparameter search and candidate selection.

```
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export RZSM_DATA_ROOT=/path/to/data_root
```

The numbered scripts can be run directly from the source tree. To inspect a script's options before preparing the inputs, run it with `--help`.

```
python CONUS_RZSM_Estimation/05_TCA_calculation.py --help
```

## Inputs

Place the prepared inputs under `Results/` inside `RZSM_DATA_ROOT`. The code uses `Results/RZSM_FNO/` when that directory exists. Set `RZSM_RESULTS_FP` or `RZSM_FIGURES_FP` to override the results or figures root. The default data root is `data/` inside the repository.

### Data availability

The source datasets below are available from their providers. This repository starts from prepared inputs and does not distribute the study's input files, model checkpoints, or results. Use the manuscript's preprocessing methods to create the study-specific grids, matched pixel time series, and cohort CSVs in a separate preprocessing workflow. The linked workflow guides identify the expected filenames and directories.

| Source dataset | Prepared input | Used by |
| --- | --- | --- |
| [ISMN station observations](https://ismn.earth/en/) (SCAN, USCRN, ARM) | Matched per-pixel NetCDF files with in situ root-zone soil moisture, satellite observations and, for the neural models, precipitation. The prepared Train and Test cohort CSVs specify which pixels to use. | Pixel EF, FNO and LSTM |
| [SMAP enhanced Level-3 soil moisture, version 6](https://nsidc.org/data/spl3smp_e/versions/6) (`SPL3SMP_E.006`) | Screened descending-pass surface soil moisture in the per-pixel files and on the common CONUS grid. | Pixel and CONUS estimation |
| [H SAF ASCAT H119 and H120](https://hsaf.meteoam.it/service/product_map.php?data_group=soil+moisture) | ASCAT surface soil moisture in the calibration pixel files and on the common CONUS grid. | EF calibration and CONUS step 02, which produces the ASCAT reference for TCA |
| [GPM IMERG Final Run V07B](https://disc.gsfc.nasa.gov/datasets/GPM_3IMERGHH_07/summary) | Daily precipitation totals for 06:00–06:00 local time in the matched pixel files and on the CONUS grid. The neural workflows accumulate these totals between SMAP observations. | Pixel FNO and LSTM, CONUS steps 03–04 |
| [MODIS Terra and Aqua vegetation indices, collection 061](https://cmr.earthdata.nasa.gov/search/site/collections/directory/LPCLOUD/gov.nasa.eosdis) (`MOD13A2` and `MYD13A2`) and [SoilGrids 2.0](https://docs.isric.org/globaldata/soilgrids/WebDav.html) | Fitting-period NDVI average and SoilGrids bulk-density layers on the study grid. | CONUS step 01, which creates the static grid used for pixel and CONUS estimation |
| [ERA5-Land](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land?tab=download) and [NLDAS-2 Noah, VIC and Mosaic](https://cmr.earthdata.nasa.gov/search/site/collections/directory/GES_DISC/gov.nasa.eosdis) | Four daily root-zone soil moisture reference grids for the 7–28 cm target, using ERA5-Land layer 2 and depth-matched NLDAS outputs. | CONUS step 05 |

The repository can generate `Model_input.nc` from the prepared static inputs, train the FNO and LSTM ensembles from the prepared pixel inputs, and produce the merged estimations read by TCA. If you already have matching checkpoints or intermediate outputs, start at the stage that reads them.

## Run

Run CONUS input preparation first. The pixel EF estimation can run with the configured SMAP timescale. Run EF step 01 separately to reproduce calibration.

```
python CONUS_RZSM_Estimation/01_Model_input.py
python Exponential_Filter/02_EF_RZSM_estimation.py
```

Prepare the neural inputs, train the ensembles, and estimate at the test pixels. The commands below use the published settings when no local Optuna selection exists. To run a new search, run all five steps in order as shown in the [FNO guide](Fourier_Neural_Operator/README.md#run-a-new-hyperparameter-search) and [LSTM guide](Long_Short-Term_Memory/README.md#run-a-new-hyperparameter-search). Step 04 uses that new selection when available. Training defaults to GPU indices 0 and 1. Add `--gpu 0` to step 04 on a single-GPU machine.

```
python Fourier_Neural_Operator/01_FNO_input_csv.py
python Fourier_Neural_Operator/04_FNO_train.py
python Fourier_Neural_Operator/05_FNO_RZSM_estimation.py

python Long_Short-Term_Memory/01_LSTM_input_csv.py
python Long_Short-Term_Memory/04_LSTM_train.py
python Long_Short-Term_Memory/05_LSTM_RZSM_estimation.py
```

Generate the CONUS estimations and run TCA after the merged estimations and land-model references are in place.

```
python CONUS_RZSM_Estimation/02_EF_RZSM_estimation.py
python CONUS_RZSM_Estimation/03_LSTM_RZSM_estimation.py
python CONUS_RZSM_Estimation/04_FNO_RZSM_estimation.py
python CONUS_RZSM_Estimation/05_TCA_calculation.py
```

## Outputs

The source uses `estimation` for model outputs and `pixel` for the evaluation unit. The renamed workflow writes CONUS results under `CONUS_RZSM_Estimation/Estimations/` and pixel results under each model's `RZSM_estimation/` directory. Estimation files end in `_estimation.nc`, and pixel-scale EF stores its series in `RZSM_estimation`.

The static grid, CONUS yearly merge and TCA can read existing CONUS inputs from the previous layout. New outputs use the current layout. Neural pixel metrics can also read series from the previous layout. To reuse other older pixel-scale exports or frozen analyses, regenerate the required products or migrate a separate copy of the results, including the corresponding NetCDF variables and provenance fields.
