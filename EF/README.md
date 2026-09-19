# Exponential-filter workflows

The public repository provides two point-scale EF workflows:

- `Fixed_EF_prediction.ipynb` calculates the primary literature baseline with `T=15`
  days for both ASCAT and SMAP. It generates Train, Test, and Excluded point
  predictions directly from prepared SSM; no precomputed EF variable is needed.
- `Calibrated_EF_prediction.ipynb` searches integer `T=1..100` separately at every
  Train/SCAN development pixel, selects one transferable value per satellite
  as the median pixel optimum, freezes it, and generates independent Test
  predictions. Test targets are never used to select T. The study data yield
  ASCAT `T=8` days and SMAP `T=10` days.

Reusable code is organized as follows:

- `filter.py`: the single observation-time recursive EF implementation shared
  by Fixed EF, Calibrated EF, and the CONUS prediction workflow.
- `calibration.py`: leakage-safe candidate scoring, pixel-optimum selection,
  sensor-level aggregation, and calibration artifact validation.
- `prediction.py`: cohort validation and point prediction export.
- `integrity.py`: safe output utilities.

Inputs are prepared point NetCDF files containing `time`, `in-situ_RZSM`, and
either `ASCAT_SSM` or `SMAP_SSM`, plus `Train_pixel_list.csv`,
`Test_pixel_list.csv`, and optionally `Excluded_pixel_list.csv` under
`Results/ISMN/Station_TCA_Screening`.

Outputs are written below `Results/EF/Fixed_EF` and
`Results/EF/Calibrated_EF`. The continental workflow uses only Fixed EF with
`T=15`; Calibrated EF is a point-scale sensitivity.
