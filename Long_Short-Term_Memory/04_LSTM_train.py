"""Train the ten-seed LSTM ensemble with the selected hyperparameters."""

import argparse
from importlib import import_module
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from core.LSTMEvent import LSTMEvent
from core.pixel_workflow import FEATURES, ModelFamily, PixelWorkflow, validate_execution
from core.rzsm_pipeline_config import MODEL_SEEDS, neural_development_metadata


def build_model(params, num_static, dynamic_channels):
    return LSTMEvent(
        hidden_size=int(params['hidden_size']), num_static_properties=num_static,
        num_layers=int(params['num_layers']), input_channels=dynamic_channels,
        include_precipitation=True, dropout_static=float(params['dropout_static']),
        dropout_fc=float(params['dropout_fc']),
    )


def suggest_parameters(trial):
    step = import_module(f"{ROOT.name}.02_LSTM_train_optuna")
    return step.suggest_parameters(trial)


FAMILY = ModelFamily('LSTM', 'Long_Short-Term_Memory', build_model, suggest_parameters)
workflow = PixelWorkflow(FAMILY, ROOT)


def prepare_ensemble_jobs(workflow, parameters, gpus):
    models = workflow.model_directory()
    models.mkdir(parents=True, exist_ok=True)
    workflow.ensemble_path().unlink(missing_ok=True)
    return [(parameters, seed, gpus[i % len(gpus)], str(models / f'Seed_{seed}'))
            for i, seed in enumerate(MODEL_SEEDS)]


def save_ensemble(workflow, selection, fits):
    pd.DataFrame(fits).to_csv(workflow.model_directory() / 'Validation.csv', index=False)
    bundle = {'format_version': 1, 'architecture': workflow.family.architecture, 'product': 'SMAP',
              'features': FEATURES, 'seeds': list(MODEL_SEEDS), 'parameters': selection['parameters'],
              'development_dates': neural_development_metadata(),
              'selected_trial': selection.get('trial'),
              'static_scaler': workflow.static_scaler().to_dict('list'), 'rain_scaler': workflow.rain_scaler(),
              'state_dicts': [torch.load(fit['checkpoint'], map_location='cpu', weights_only=True) for fit in fits]}
    temporary = workflow.ensemble_path().with_suffix('.pt.tmp')
    torch.save(bundle, temporary)
    temporary.replace(workflow.ensemble_path())


def build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--gpu', nargs='+', type=int, default=[0, 1], help='GPU IDs (default: 0 1).')
    parser.add_argument('--workers', type=int, default=10, help='Concurrent seed fits (default: 10).')
    return parser


if __name__ == '__main__':
    workflow = import_module('Long_Short-Term_Memory.04_LSTM_train').workflow
    args = build_argument_parser().parse_args()
    selection = workflow.load_parameters()
    validate_execution(args.gpu, args.workers)
    jobs = prepare_ensemble_jobs(workflow, selection['parameters'], args.gpu)
    fits = workflow.run_jobs(jobs, args.workers, epoch_bars=[f'Seed {seed:>3}' for seed in MODEL_SEEDS])
    save_ensemble(workflow, selection, fits)
    print("04_LSTM_train.py is finished.")
