"""Shared paths, fitting, and output routines for pixel-scale models."""
from __future__ import annotations

import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, wait, as_completed, FIRST_COMPLETED
from dataclasses import dataclass
from pathlib import Path
from queue import Empty
from typing import Callable

import numpy as np
import pandas as pd
import netCDF4 as nc
import torch
from tqdm import tqdm

try:
    import optuna
except ModuleNotFoundError as error:
    if error.name != "optuna":
        raise
    optuna = None

from core import pixel_training as training
from core.precipitation_inputs import (
    RAINFALL_FILE, event_interval_amounts, event_windows, precipitation_windows,
    require_rainfall_file, standardized_precipitation_windows,
)
from core.paper_parameters import paper_selection
from core.pixel_seed_diagnostics import (
    calculate_estimation_metrics, read_seed_ensemble, summarize_seed_spread,
)
from core.project_paths import existing_result_path, PATHS
from core.rzsm_pipeline_config import (
    MODEL_SEED, MODEL_SEEDS, build_observation_event_sequences, dates_from_variable,
    neural_development_metadata,
    pixel_context_period_indices, split_boundaries, test_period_mask, test_period_metadata,
)
from core.static_input_profiles import EVENT_FEATURES

AREAS = ('Train', 'Test')
FEATURES = list(EVENT_FEATURES)
METADATA_COLUMNS = ['lat_idx', 'lon_idx', 'lat', 'lon', 'valid_SSM', 'valid_RZSM']
FIXED_PARAMETERS = dict(batch_size=1024, eta_min=1e-6, scheduler_factor=.5, scheduler_patience=10)
RAIN_LAG_DAYS = 1
ESTIMATION_BATCH = 1024


_ESTIMATION = None


@dataclass(frozen=True)
class ModelFamily:
    name: str
    folder: str
    build_model: Callable
    suggest: Callable
    estimation_attributes: Callable | None = None
    label: str | None = None

    @property
    def architecture(self):
        """Label stored in the ensemble file ('full' for FNO, 'LSTM' for LSTM)."""
        return self.label or self.name


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def read_pixel_precipitation(dataset, dates):
    """Daily rainfall (mm) from a pixel file, aligned to ``dates``."""
    if 'Precipitation' not in dataset.variables:
        raise KeyError('Pixel NetCDF is missing Precipitation (daily totals in mm)')
    var = dataset['Precipitation']
    units = str(getattr(var, 'units', '')).lower().replace(' ', '')
    if units not in ('mm', 'mm/day', 'mmd-1', 'mmday-1'):
        raise ValueError(f'Expected daily pixel precipitation in mm; received {units!r}')
    source_dates = dates_from_variable(dataset['time'])
    indices = np.searchsorted(source_dates, dates)
    if (indices >= len(source_dates)).any() or not np.array_equal(source_dates[indices], dates):
        raise ValueError('Pixel precipitation dates do not match the SMAP dates')
    return np.asarray(np.ma.filled(var[:], np.nan), dtype=np.float32).reshape(-1)[indices]


def pixel_rainfall_file(dataset):
    """Name of the gridded rain file a pixel file's Precipitation was taken from."""
    return os.path.basename(str(getattr(dataset['Precipitation'], 'source_file', '')))


def validate_execution(gpus, workers):
    if workers < 1 or not gpus or min(gpus) < 0 or len(gpus) != len(set(gpus)):
        raise ValueError('Use positive workers and distinct nonnegative GPU IDs')
    if torch.cuda.is_available() and max(gpus) >= torch.cuda.device_count():
        raise ValueError(f'Available CUDA devices: 0 through {torch.cuda.device_count() - 1}')


def candidate_parameters(trial):
    return {**FIXED_PARAMETERS, **trial.params}


def require_optuna():
    if optuna is None:
        raise ModuleNotFoundError("Install optuna to search or review hyperparameter trials.", name="optuna")
    return optuna


def best_distinct_trials(study, top_trials):
    """Take the lowest-loss completed trial for each distinct parameter set."""
    optuna = require_optuna()

    candidates, seen = [], set()
    complete = (t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
    for trial in sorted(complete, key=lambda t: (t.value, t.number)):
        key = json.dumps(trial.params, sort_keys=True)
        if key not in seen:
            seen.add(key)
            candidates.append(trial)
        if len(candidates) == top_trials:
            break
    if len(candidates) < top_trials:
        raise ValueError(f'Only {len(candidates)} distinct completed trials; requested {top_trials}')
    return candidates


def rank_candidates(seed_results):
    """Rank trials by mean combined loss, then mean spatial loss, seed SD and trial number."""
    rows = []
    for trial, group in seed_results.groupby('trial', sort=True):
        if (sorted(group.seed) != sorted(MODEL_SEEDS)
                or not np.isfinite(group[['combined_loss', 'spatial_loss']].to_numpy()).all()):
            raise ValueError(f'Candidate {trial} needs one finite result for every seed')
        rows.append({'trial': int(trial), 'mean_validation_loss': float(group.combined_loss.mean()),
                     'mean_spatial_validation_loss': float(group.spatial_loss.mean()),
                     'std_validation_loss': float(group.combined_loss.std(ddof=1)) if len(group) > 1 else 0.})
    return pd.DataFrame(rows).sort_values(
        ['mean_validation_loss', 'mean_spatial_validation_loss', 'std_validation_loss', 'trial']
    ).reset_index(drop=True)


class PixelWorkflow:
    def __init__(self, family: ModelFamily, source_dir):
        self.family = family
        self.source_dir = Path(source_dir)
        self.results = PATHS.results / family.folder
        self.figures = PATHS.figures / family.folder
        self.inputs = self.results / 'Inputs'
        self.ismn = PATHS.results / 'International_Soil_Moisture_Network'
        self.optuna = self.results / 'Optuna'


    def input_csv(self, area):
        if area not in AREAS:
            raise ValueError(f'Unknown area: {area}')
        return self.inputs / area / f'{self.family.name}_input_SMAP.csv'


    def model_directory(self):
        return self.results / 'Models'


    def ensemble_path(self):
        return self.model_directory() / f'{self.family.name}_ensemble.pt'


    def estimation_directory(self, area):
        if area not in AREAS:
            raise ValueError(f'Unknown area: {area}')
        return self.results / 'RZSM_estimation' / area


    def read_table(self, area):
        path = self.input_csv(area)
        frame = pd.read_csv(path)
        if list(frame.columns) != METADATA_COLUMNS + FEATURES:
            raise ValueError(f'{path}: expected columns {METADATA_COLUMNS + FEATURES}; run 01 again')
        if frame.empty:
            raise ValueError(f'{path}: required {area} inputs are empty; run 01 again')
        if frame.duplicated(['lat_idx', 'lon_idx']).any() or not np.isfinite(frame[FEATURES].to_numpy()).all():
            raise ValueError(f'{path}: duplicate pixels or invalid static inputs')
        return frame


    def training_table(self):
        frame = self.read_table('Train')
        if len(frame) < 2:
            raise ValueError('Need at least two training pixels for spatial validation')
        return frame


    def prepare_training_inputs(self):
        """Write the static scaler and per-pixel rainfall intervals for training."""
        table = self.training_table()
        self.write_static_scaler(table)
        return self.prepare_rainfall(table)


    def write_static_scaler(self, table):
        self.inputs.mkdir(parents=True, exist_ok=True)
        training.fit_feature_scaler(table, FEATURES).to_csv(self.inputs / 'Static_scaler.csv', index=False)


    def prepare_rainfall(self, table):
        """Save rainfall between SMAP observations and fit its scaler on training windows.

        The interval ending at observation b holds the daily totals (06:00-06:00
        local days) from the previous observation a through day b-1; rain on
        day b is excluded. Only dates before the test period are read.
        """
        destination = self.inputs / 'Rain'
        destination.mkdir(parents=True, exist_ok=True)
        train_pixels = training.spatial_training_pixels(len(table))
        coverage, training_values = [], []
        for position, row in enumerate(table.itertuples()):
            pixel = f'{int(row.lat_idx)}_{int(row.lon_idx)}'
            with nc.Dataset(self.ismn / 'ISMN_Train' / f'{pixel}.nc') as ds:
                ssm, y, dates, start, _ = training.read_pixel_training_series(ds)
                daily = read_pixel_precipitation(ds, dates)
                require_rainfall_file(pixel_rainfall_file(ds), f'{pixel} pixel file')
            train_stop, val_stop = split_boundaries(dates)
            ssm, y, dates, daily = ssm[:val_stop], y[:val_stop], dates[:val_stop], daily[:val_stop]
            y = training.mask_to_ssm_observations(y, ssm)
            intervals = event_interval_amounts(ssm, dates, dates, daily, RAIN_LAG_DAYS)
            seen = position in train_pixels
            partitions = ({'train': (start, train_stop), 'temporal_validation': (train_stop, val_stop)}
                          if seen else {'spatial_validation': (start, val_stop)})
            record = {'pixel': pixel, 'spatial_partition': 'training' if seen else 'validation',
                      'train_windows': 0, 'temporal_validation_windows': 0, 'spatial_validation_windows': 0}
            for split, (lo, hi) in partitions.items():
                _, _, indices = build_observation_event_sequences(ssm, y, time_values=dates,
                                                                  target_start=lo, target_stop=hi)
                try:
                    precipitation_windows(ssm, indices, intervals)
                except ValueError as error:
                    raise ValueError(f'{pixel}, {split}: {error}') from error
                record[split + '_windows'] = len(indices)
                if split == 'train' and len(indices):
                    used = np.unique(event_windows(ssm, indices)[:, 1:])
                    training_values.append(np.log1p(intervals[used].astype(np.float64)))
            coverage.append(record)
            np.savez_compressed(destination / f'{pixel}.npz', dates=dates, interval_mm=intervals)
        if not training_values:
            raise ValueError('No training precipitation intervals')
        values = np.concatenate(training_values)
        scale = float(values.std(ddof=0))
        scaler = {'mean': float(values.mean()), 'std': scale if scale > 1e-8 else 1.,
                  'transform': 'log1p(interval_mm)', 'training_interval_count': len(values),
                  'rainfall_file': RAINFALL_FILE}
        pd.DataFrame([scaler]).to_csv(destination / 'Scaler.csv', index=False)
        pd.DataFrame(coverage).to_csv(destination / 'Coverage.csv', index=False)
        return scaler


    def static_scaler(self):
        return pd.read_csv(self.inputs / 'Static_scaler.csv')


    def rain_scaler(self):
        row = pd.read_csv(self.inputs / 'Rain' / 'Scaler.csv').iloc[0]
        scaler = {'mean': float(row['mean']), 'std': float(row['std'])}
        if not np.isfinite(list(scaler.values())).all() or scaler['std'] <= 0:
            raise ValueError('Invalid rainfall scaler; run 01 again')
        rainfall_file = row.get('rainfall_file')
        require_rainfall_file(rainfall_file if isinstance(rainfall_file, str) else '', 'Rain/Scaler.csv (run 01 again)')
        scaler['rainfall_file'] = rainfall_file
        return scaler


    def fit_job(self, job, report=None):
        """Fit one (parameters, seed, gpu, folder) job in a worker process.

        report=(queue, key) sends (key, epoch, validation loss) after every epoch.
        """
        params, seed, gpu, folder = job
        folder = Path(folder)
        relative = folder.relative_to(self.results) if folder.is_relative_to(self.results) else Path(folder.name)
        on_epoch = None
        if report is not None:
            queue, key = report
            on_epoch = lambda epoch, loss: queue.put((key, epoch, loss))
        return training.train_one_model(
            self.family.build_model, self.training_table(), FEATURES, self.static_scaler(),
            {'input_dir': str(self.inputs / 'Rain'), 'scaler': self.rain_scaler()},
            str(self.ismn / 'ISMN_Train'), str(folder), gpu, seed, params,
            loss_plot=str(self.figures / relative / f'Seed_{seed}_loss_trend.png'), on_epoch=on_epoch)


    def run_jobs(self, jobs, workers, progress=None, epoch_bars=None):
        """Run fits in spawned processes; epoch_bars draws one live bar per job."""
        if not jobs:
            return []
        context = mp.get_context('spawn')
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs)), mp_context=context) as pool:
            if epoch_bars is not None:
                return self._run_with_epoch_bars(pool, context, jobs, epoch_bars)
            futures = [pool.submit(self.fit_job, job) for job in jobs]
            for future in as_completed(futures):
                future.result()
                if progress is not None:
                    progress.update()
            return [future.result() for future in futures]


    def _run_with_epoch_bars(self, pool, context, jobs, labels):
        with context.Manager() as manager:
            queue = manager.Queue()
            bars = [tqdm(total=training.DEFAULT_NUM_EPOCHS, desc=label, unit='epoch', position=i,
                         leave=False, dynamic_ncols=True) for i, label in enumerate(labels)]
            try:
                futures = [pool.submit(self.fit_job, job, (queue, i)) for i, job in enumerate(jobs)]
                pending = set(futures)
                while pending:
                    done, pending = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
                    while True:
                        try:
                            i, epoch, loss = queue.get_nowait()
                        except Empty:
                            break
                        bars[i].update(epoch - bars[i].n)
                        bars[i].set_postfix_str(f'val {loss:.4g}', refresh=False)
                    for future in done:
                        future.result()
                return [future.result() for future in futures]
            finally:
                for bar in reversed(bars):
                    bar.close()


    def open_study(self, directory=None, study_name=None):
        optuna = require_optuna()
        directory = Path(directory or self.optuna)
        if not (self.inputs / 'Rain' / 'Scaler.csv').is_file():
            raise FileNotFoundError(f'{self.inputs}: run 01 first')
        directory.mkdir(parents=True, exist_ok=True)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        database = (directory / f'{self.family.name}_SMAP.db').resolve()
        study = optuna.create_study(
            study_name=study_name or f'SMAP_{self.family.name}', direction='minimize',
            storage=f'sqlite:///{database.as_posix()}', load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=MODEL_SEED), pruner=optuna.pruners.NopPruner())
        recorded = study.user_attrs.get('static_features')
        if (recorded is None and study.trials) or recorded not in (None, FEATURES):
            raise ValueError(f'{database}: study was run with static features {recorded or "unrecorded (NDVI)"}, '
                             f'not {FEATURES}; move it aside to search again')
        study.set_user_attr('static_features', FEATURES)
        return study


    def tune(self, gpus, workers=20, trials=200, *, directory=None, suggest=None, study_name=None):
        """Run a single-seed search until ``trials`` trials exist in the study."""
        validate_execution(gpus, workers)
        directory = Path(directory or self.optuna)
        suggest = suggest or (lambda trial: {**FIXED_PARAMETERS, **self.family.suggest(trial)})
        study = self.open_study(directory, study_name)
        self.fit_optuna_trials(study, directory, suggest, gpus, workers, trials)
        study.trials_dataframe().to_csv(directory / 'Trials.csv', index=False)
        return study


    def fit_optuna_trials(self, study, directory, suggest, gpus, workers, trials):
        """Fill the remaining trial slots with single-seed fits."""
        optuna = require_optuna()
        directory.mkdir(parents=True, exist_ok=True)
        running = (optuna.trial.TrialState.RUNNING, optuna.trial.TrialState.WAITING)
        if any(t.state in running for t in study.trials):
            raise ValueError('The study has unfinished trials; finish or remove them before resuming')
        remaining = max(0, trials - len(study.trials))
        if remaining:
            with ProcessPoolExecutor(max_workers=min(workers, remaining), mp_context=mp.get_context('spawn')) as pool, \
                    tqdm(total=trials, initial=trials - remaining, desc='Optuna trials', unit='trial', smoothing=0) as progress:
                pending, launched = {}, 0
                try:
                    while remaining or pending:
                        while remaining and len(pending) < workers:
                            trial = study.ask()
                            job = (suggest(trial), MODEL_SEED, gpus[launched % len(gpus)],
                                   str(directory / 'Trials' / f'Trial_{trial.number}'))
                            pending[pool.submit(self.fit_job, job)] = trial
                            launched += 1
                            remaining -= 1
                        ready, _ = wait(pending, return_when=FIRST_COMPLETED)
                        for future in ready:
                            trial = pending.pop(future)
                            study.tell(trial, future.result()['combined_loss'])
                            progress.update()
                except BaseException:
                    for future, trial in pending.items():
                        future.cancel()
                        study.tell(trial, state=optuna.trial.TrialState.FAIL, skip_if_finished=True)
                    raise


    def review_candidates(self, gpus, workers=40, top_trials=10, *, directory=None, parameters=None,
                          study_name=None):
        """Refit the best distinct trials with every seed and select by mean validation loss."""
        validate_execution(gpus, workers)
        directory = Path(directory or self.optuna)
        parameters = parameters or candidate_parameters
        study = self.open_study(directory, study_name)
        candidates = best_distinct_trials(study, top_trials)
        seed_results = self.refit_candidates(candidates, directory, parameters, gpus, workers)
        ranking = rank_candidates(seed_results)
        self.save_candidate_review(directory, candidates, seed_results, ranking, parameters)


    def refit_candidates(self, candidates, directory, parameters, gpus, workers):
        """Fit each candidate with every ensemble seed and collect validation losses."""
        pairs = [(trial, seed) for trial in candidates for seed in MODEL_SEEDS]
        jobs = [(parameters(trial), seed, gpus[i % len(gpus)],
                 str(directory / 'Candidates' / f'Trial_{trial.number}' / f'Seed_{seed}'))
                for i, (trial, seed) in enumerate(pairs)]
        with tqdm(total=len(jobs), desc='Candidate models', unit='model', smoothing=0) as progress:
            fits = self.run_jobs(jobs, workers, progress)
        return pd.DataFrame([{'trial': trial.number, **fit} for (trial, _), fit in zip(pairs, fits)])


    def save_candidate_review(self, directory, candidates, seed_results, ranking, parameters):
        seed_results.to_csv(directory / 'Candidate_seed_validation.csv', index=False)
        ranking.to_csv(directory / 'Candidate_ranking.csv', index=False)
        winner = next(trial for trial in candidates if trial.number == int(ranking.trial.iloc[0]))
        write_json(directory / 'Best_parameters.json', {
            'family': self.family.name, 'architecture': self.family.architecture,
            'trial': winner.number, 'parameters': parameters(winner)})
        print(f'Selected trial {winner.number}; parameters written to {directory / "Best_parameters.json"}')


    def load_parameters(self, path=None):
        if path is None:
            path = self.optuna / 'Best_parameters.json'
            if not path.exists():
                return paper_selection(self.family.name)
        path = Path(path)
        selection = json.loads(path.read_text())
        if selection.get('family') != self.family.name:
            raise ValueError(f'{path} holds {selection.get("family")} parameters, not {self.family.name}')
        return selection


    def train_final(self, gpus, workers=10, parameters_path=None):
        """Train every seed with the selected parameters and save one ensemble file."""
        selection = self.load_parameters(parameters_path)
        self.train_selected_ensemble(selection, gpus, workers)


    def train_selected_ensemble(self, selection, gpus, workers):
        """Fit and save the ensemble for an already loaded parameter selection."""
        validate_execution(gpus, workers)
        jobs = self.prepare_ensemble_jobs(selection['parameters'], gpus)
        fits = self.run_jobs(jobs, workers, epoch_bars=[f'Seed {seed:>3}' for seed in MODEL_SEEDS])
        self.save_ensemble(selection, fits)


    def prepare_ensemble_jobs(self, parameters, gpus):
        models = self.model_directory()
        models.mkdir(parents=True, exist_ok=True)
        self.ensemble_path().unlink(missing_ok=True)
        return [(parameters, seed, gpus[i % len(gpus)], str(models / f'Seed_{seed}'))
                for i, seed in enumerate(MODEL_SEEDS)]


    def save_ensemble(self, selection, fits):
        pd.DataFrame(fits).to_csv(self.model_directory() / 'Validation.csv', index=False)
        bundle = {'format_version': 1, 'architecture': self.family.architecture, 'product': 'SMAP',
                  'features': FEATURES, 'seeds': list(MODEL_SEEDS), 'parameters': selection['parameters'],
                  'development_dates': neural_development_metadata(),
                  'selected_trial': selection.get('trial'),
                  'static_scaler': self.static_scaler().to_dict('list'), 'rain_scaler': self.rain_scaler(),
                  'state_dicts': [torch.load(fit['checkpoint'], map_location='cpu', weights_only=True) for fit in fits]}
        temporary = self.ensemble_path().with_suffix('.pt.tmp')
        torch.save(bundle, temporary)
        temporary.replace(self.ensemble_path())


    def load_ensemble(self):
        path = self.ensemble_path()
        bundle = torch.load(path, map_location='cpu', weights_only=False)
        if (bundle['architecture'] != self.family.architecture or bundle['features'] != FEATURES
                or len(bundle['state_dicts']) != len(bundle['seeds'])):
            raise ValueError(f'{path} does not match the {self.family.name} inputs')
        models = []
        for state in bundle['state_dicts']:
            model = self.family.build_model(bundle['parameters'], len(FEATURES), 3)
            model.load_state_dict(state)
            model.eval()
            models.append(model)
        return bundle, models


    def _initialize_estimation(self):
        global _ESTIMATION
        torch.set_num_threads(1)
        _ESTIMATION = self.load_ensemble()


    def estimate_pixel(self, task):
        area, row = task
        bundle, models = _ESTIMATION
        pixel = f"{int(row['lat_idx'])}_{int(row['lon_idx'])}"
        with nc.Dataset(self.ismn / f'ISMN_{area}' / f'{pixel}.nc') as ds:
            ssm, obs, dates, start, stop = training.read_pixel_training_series(ds)
            daily_rain = read_pixel_precipitation(ds, dates)
            require_rainfall_file(pixel_rainfall_file(ds), f'{area}/{pixel} pixel file')
            indices, _, _ = pixel_context_period_indices(dates_from_variable(ds['time']), require_complete=True)
            raw_time = np.asarray(ds['time'][:])[indices]
            units, calendar = ds['time'].units, getattr(ds['time'], 'calendar', 'standard')
            ef = (np.ma.filled(ds['SMAP_RZSM'][:], np.nan).reshape(-1)[indices]
                  if 'SMAP_RZSM' in ds.variables else np.full(len(dates), np.nan))
        windows, _, targets = build_observation_event_sequences(ssm, time_values=dates, target_start=start, target_stop=stop)
        if not len(targets):
            return pixel, False
        require_rainfall_file(bundle['rain_scaler'].get('rainfall_file', ''), 'ensemble rain scaler (retrain 01-04)')
        intervals = event_interval_amounts(ssm, dates, dates, daily_rain, RAIN_LAG_DAYS)
        rain = standardized_precipitation_windows(ssm, targets, intervals, bundle['rain_scaler'])
        dynamic = np.concatenate([windows, rain[..., None]], axis=-1)
        frame = pd.DataFrame([row])
        training.apply_feature_scaler(frame, pd.DataFrame(bundle['static_scaler']))
        static = torch.tensor(frame[FEATURES].to_numpy(), dtype=torch.float32)
        estimations = np.full((len(models), len(dates)), np.nan, dtype=np.float32)
        with torch.inference_mode():
            for i, model in enumerate(models):
                for offset in range(0, len(targets), ESTIMATION_BATCH):
                    x = torch.tensor(dynamic[offset:offset + ESTIMATION_BATCH], dtype=torch.float32)
                    estimations[i, targets[offset:offset + len(x)]] = model(x, static.expand(len(x), -1)).numpy().reshape(-1)
        mean = np.full(len(dates), np.nan, dtype=np.float32)
        spread = mean.copy()
        mean[targets] = estimations[:, targets].mean(axis=0)
        spread[targets] = estimations[:, targets].std(axis=0, ddof=1)
        output = self.estimation_directory(area) / f'{pixel}_estimation.nc'
        temp = output.with_name(output.name + f'.tmp.{os.getpid()}')
        try:
            with nc.Dataset(temp, 'w') as ds:
                ds.createDimension('time', len(dates))
                t = ds.createVariable('time', 'i4', ('time',))
                t[:] = raw_time
                t.units, t.calendar = units, calendar
                values = dict(SMAP_SSM=ssm, RZSM=obs, EF_RZSM=ef, RZSM_avg=mean, RZSM_std=spread)
                values.update({f'RZSM_trial_{i + 1}': p for i, p in enumerate(estimations)})
                for name, value in values.items():
                    v = ds.createVariable(name, 'f4', ('time',))
                    v[:] = value
                    v.units = 'm3 m-3'
                    if name.startswith('RZSM_trial_'):
                        v.seed = bundle['seeds'][int(name.rsplit('_', 1)[1]) - 1]
                ds['RZSM_std'].ddof = 1
                ds.trial_count = len(models)
                ds.static_features = ','.join(bundle['features'])
                if self.family.estimation_attributes is not None:
                    ds.setncatts(self.family.estimation_attributes(models))
                ds.lat, ds.lon = row['lat'], row['lon']
            temp.replace(output)
        finally:
            temp.unlink(missing_ok=True)
        return pixel, True


    def estimate(self, areas, workers=40):
        if workers < 1:
            raise ValueError('workers must be positive')
        self.load_ensemble()
        for area in areas:
            pixels = self.read_table(area)
            results = self.estimate_pixels(area, pixels, workers)
            self.finish_estimations(area, results)


    def estimate_pixels(self, area, pixels, workers):
        """Run ensemble inference for the rows in a pixel table."""
        self.estimation_directory(area).mkdir(parents=True, exist_ok=True)
        tasks = [(area, row) for row in pixels.to_dict('records')]
        with ProcessPoolExecutor(max_workers=min(workers, len(tasks)), mp_context=mp.get_context('spawn'),
                                 initializer=self._initialize_estimation) as pool:
            return list(tqdm(pool.map(self.estimate_pixel, tasks), total=len(tasks),
                             desc=f'{area} pixels', unit='pixel', leave=False))


    def finish_estimations(self, area, results):
        saved = {pixel for pixel, success in results if success}
        folder = self.estimation_directory(area)
        for path in folder.glob('*_estimation.nc'):
            if path.name.removesuffix('_estimation.nc') not in saved:
                path.unlink()
        print(f'{area}: saved {len(saved)} of {len(results)} pixels '
              f'({len(results) - len(saved)} lack a complete event history)')


    def estimation_files(self, area):
        """Estimation files of the input pixels, in input-table order."""
        folder = self.estimation_directory(area)
        read_legacy = not folder.exists()
        paths = []
        for row in self.read_table(area).itertuples():
            path = folder / f'{int(row.lat_idx)}_{int(row.lon_idx)}_estimation.nc'
            if read_legacy:
                path = existing_result_path(path)
            if path.is_file():
                paths.append(path)
        if not paths:
            raise FileNotFoundError(f'{folder}: no estimations; run 05 first')
        return paths


    def evaluate(self, area, progress=False):
        """Score finite observed/ensemble pairs on test dates (EF is not used)."""
        rows, pooled_obs, pooled_pred = [], [], []
        for path in tqdm(self.estimation_files(area), desc=f'{area} metrics', unit='pixel',
                         leave=False, disable=not progress):
            with nc.Dataset(path) as ds:
                mask = test_period_mask(dates_from_variable(ds['time']))
                obs = np.ma.filled(ds['RZSM'][:], np.nan)[mask]
                ensemble = read_seed_ensemble(ds)
                mean, std = ensemble['ensemble_mean'][mask], ensemble['seed_spread'][mask]
                valid = np.isfinite(obs) & np.isfinite(mean)
                pooled_obs.extend(obs[valid])
                pooled_pred.extend(mean[valid])
                y, x = map(int, path.name.split('_')[:2])
                row = {'lat_idx': y, 'lon_idx': x, 'lat': ds.lat, 'lon': ds.lon, **test_period_metadata(),
                       **{f'{k}_avg': v for k, v in calculate_estimation_metrics(obs, mean).items()},
                       **summarize_seed_spread(obs, mean, std)}
                for i, (seed, values) in enumerate(zip(ensemble['seeds'], ensemble['trials']), 1):
                    row[f'model_seed_trial_{i}'] = seed
                    row.update({f'{k}_trial_{i}': v for k, v in calculate_estimation_metrics(obs, values[mask]).items()})
                rows.append(row)
        return pd.DataFrame(rows), np.asarray(pooled_obs), np.asarray(pooled_pred)


    def calculate_metrics(self, areas=('Test',)):
        for area in areas:
            frame, _, _ = self.evaluate(area, progress=True)
            folder = self.estimation_directory(area).parent
            folder.mkdir(parents=True, exist_ok=True)
            frame.to_csv(folder / f'{area}_metrics.csv', index=False)
