# Exponential filter workflow

`EF_preprocessing.ipynb` is the complete executable workflow. It preserves the
behavior of the three active scripts in the older `Exponential_Filter` folder
while keeping products, point cohorts, dates, fixed-filter settings, paths,
loops, multiprocessing, and output inspection visible in the notebook.
Its main sections follow the legacy order: fixed-T prediction export
(`01_EF_calculation.py`), pooled scatterplots (`02_EF_scatterplot.py`), then
point metrics and timeseries figures (`03_EF_timeseries_plot.py`). In the last
section, run the metric-CSV cell first. The following plotting cell reads and
validates those saved CSVs without recalculating or overwriting them, so figure
generation can be run or resumed separately.

The reusable helpers are organized as follows:

- `prediction.py`: canonical-cohort validation, export of the fixed-T RZSM
  series embedded in each common ISMN point file, and station metrics.
- `evaluation.py`: period selection, prediction-file reading, and paired RZSM
  metrics.
- `plotting.py`: pooled test-period density scatterplots and station
  timeseries figures.
- `integrity.py`: atomic CSV writes and narrow output cleanup.

The workflow uses the fixed T = 15 day ASCAT and SMAP RZSM series already
stored by `ISMN_preprocessing.ipynb`; it does not recalculate or optimize T.
Those series update only on valid SSM observation dates, remain `NaN` on
non-observation dates, and begin after the one-calendar-year adjustment ending
2016-03-31. Every point input must have the exact 3,197 daily timestamps from
2015-04-01 through 2023-12-31; legacy 3,198-day inputs beginning 2015-03-31 are
rejected before prediction export.
The common point cohort is read from the existing FNO input CSV for each
area/product pair. Train, Test, and Excluded pixels are included by default in
prediction export, scatterplots, metric CSVs, and timeseries figures.
Test-period comparisons use the final 20% of the post-spin-up common daily
record, 2022-06-13 through 2023-12-31 inclusive. Test metric CSVs record this
policy, the 0.80 start ratio, and both boundary dates.

Numerical outputs are written under the repository-local `Results/EF`
location, and rendered diagnostics are written under `Outputs/Figures/EF`.
Run
the ISMN workflow and create the canonical FNO cohort CSVs before running this
notebook.
