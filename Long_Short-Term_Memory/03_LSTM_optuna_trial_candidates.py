"""Refit the best distinct trials of step 02 with all ten seeds and select one.

Writes Results/Long_Short-Term_Memory/Optuna/Best_parameters.json for step 04.
"""

import argparse
import json
from importlib import import_module
from pathlib import Path
import sys

import optuna
import pandas as pd
from tqdm import tqdm

publication_root = Path(__file__).resolve().parents[1]
if str(publication_root) not in sys.path:
    sys.path.insert(0, str(publication_root))
workflow = import_module('Long_Short-Term_Memory.04_LSTM_train').workflow

from core.pixel_workflow import (
    candidate_parameters,
    rank_candidates,
    validate_execution,
    write_json,
)
from core.rzsm_pipeline_config import MODEL_SEEDS


def best_distinct_trials(study, top_trials):
    """Take the lowest-loss completed trial for each distinct parameter set."""
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


def refit_candidates(workflow, candidates, directory, parameters, gpus, workers):
    """Fit each candidate with every ensemble seed and collect validation losses."""
    pairs = [(trial, seed) for trial in candidates for seed in MODEL_SEEDS]
    jobs = [(parameters(trial), seed, gpus[i % len(gpus)],
             str(directory / 'Candidates' / f'Trial_{trial.number}' / f'Seed_{seed}'))
            for i, (trial, seed) in enumerate(pairs)]
    with tqdm(total=len(jobs), desc='Candidate models', unit='model', smoothing=0) as progress:
        fits = workflow.run_jobs(jobs, workers, progress)
    return pd.DataFrame([{'trial': trial.number, **fit} for (trial, _), fit in zip(pairs, fits)])


def save_candidate_review(workflow, directory, candidates, seed_results, ranking, parameters):
    seed_results.to_csv(directory / 'Candidate_seed_validation.csv', index=False)
    ranking.to_csv(directory / 'Candidate_ranking.csv', index=False)
    winner = next(trial for trial in candidates if trial.number == int(ranking.trial.iloc[0]))
    write_json(directory / 'Best_parameters.json', {
        'family': workflow.family.name, 'architecture': workflow.family.architecture,
        'trial': winner.number, 'parameters': parameters(winner)})
    print(f'Selected trial {winner.number}; parameters written to {directory / "Best_parameters.json"}')


def build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--gpu', nargs='+', type=int, default=[0, 1], help='GPU IDs (default: 0 1).')
    parser.add_argument('--workers', type=int, default=40, help='Concurrent fits (default: 40).')
    return parser


if __name__ == '__main__':
    args = build_argument_parser().parse_args()
    validate_execution(args.gpu, args.workers)
    directory = workflow.optuna
    study = workflow.open_study(directory)
    candidates = best_distinct_trials(study, top_trials=10)
    seed_results = refit_candidates(workflow, candidates, directory, candidate_parameters, args.gpu, args.workers)
    ranking = rank_candidates(seed_results)
    save_candidate_review(workflow, directory, candidates, seed_results, ranking, candidate_parameters)
    print("03_LSTM_optuna_trial_candidates.py is finished.")
