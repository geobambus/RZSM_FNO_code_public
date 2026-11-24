# core

Code shared by the model folders.

## Setup

The numbered scripts load `core` from this source tree. See the [repository README](../README.md) for setup and run commands.

## Modules

| Module | Purpose |
|---|---|
| `project_paths.py` | Data root (`RZSM_DATA_ROOT`) and the `Results/`/`Figures/` folders |
| `rzsm_pipeline_config.py` | Dates, train/validation/test split, 32-event observation windows, EF settings |
| `pixel_training.py` | Fitting one FNO or LSTM seed: windows, loss, early stopping, optimizer |
| `pixel_workflow.py` | Shared pixel-model fitting, configuration and ensemble I/O |
| `static_input_profiles.py` | Static predictor order (NDVI, bulk density) and grid schema |
| `precipitation_inputs.py` | Rainfall between SMAP observations (06:00-06:00 local days) and its window channel |
| `pixel_seed_diagnostics.py` | Reading ten-seed estimations and pixel metrics |
| `paper_parameters.py` | Published FNO and LSTM training selections used when step 03 has no result |
| `FNOEvent.py`, `LSTMEvent.py` | The two networks |
