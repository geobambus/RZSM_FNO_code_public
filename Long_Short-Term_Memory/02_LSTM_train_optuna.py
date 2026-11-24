"""Run a new single-seed Optuna search for the LSTM (writes Results/Long_Short-Term_Memory/Optuna).

Optional: step 04 can train with the published settings in core/paper_parameters.py.
"""

import argparse
import multiprocessing as mp
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from importlib import import_module
from pathlib import Path
import sys

import optuna
from tqdm import tqdm

publication_root = Path(__file__).resolve().parents[1]
if str(publication_root) not in sys.path:
    sys.path.insert(0, str(publication_root))

workflow = import_module('Long_Short-Term_Memory.04_LSTM_train').workflow

from core.pixel_workflow import FIXED_PARAMETERS, validate_execution
from core.rzsm_pipeline_config import MODEL_SEED


HIDDEN_SIZES = (32, 64, 128)


def suggest_parameters(trial):
    return {'learning_rate': trial.suggest_float('learning_rate', 5e-5, 2e-3, log=True),
            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 5e-2, log=True),
            'hidden_size': trial.suggest_categorical('hidden_size', list(HIDDEN_SIZES)),
            'num_layers': trial.suggest_int('num_layers', 1, 3),
            'dropout_static': trial.suggest_float('dropout_static', .05, .70),
            'dropout_fc': trial.suggest_float('dropout_fc', .05, .70)}


def fit_optuna_trials(workflow, study, directory, suggest, gpus, workers, trials):
    """Fill the remaining trial slots with single-seed fits."""
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
                        pending[pool.submit(workflow.fit_job, job)] = trial
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


def build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--gpu', nargs='+', type=int, default=[0, 1], help='GPU IDs (default: 0 1).')
    parser.add_argument('--workers', type=int, default=20, help='Concurrent fits (default: 20).')
    return parser


if __name__ == '__main__':
    args = build_argument_parser().parse_args()
    validate_execution(args.gpu, args.workers)
    directory = workflow.optuna
    study = workflow.open_study(directory)

    def trial_parameters(trial):
        return {**FIXED_PARAMETERS, **suggest_parameters(trial)}

    fit_optuna_trials(workflow, study, directory, trial_parameters, args.gpu, args.workers, trials=200)
    study.trials_dataframe().to_csv(directory / 'Trials.csv', index=False)
    print("02_LSTM_train_optuna.py is finished.")
