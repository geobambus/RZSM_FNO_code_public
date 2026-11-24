"""Train the ten-seed FNO ensemble with the selected hyperparameters."""

import argparse
from importlib import import_module
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from core.FNOEvent import FNOEvent
from core.pixel_workflow import FEATURES, ModelFamily, PixelWorkflow, validate_execution
from core.rzsm_pipeline_config import MODEL_SEEDS, OBSERVATION_EVENT_SEQUENCE_LENGTH, neural_development_metadata


def build_model(params, num_static, dynamic_channels):

    return FNOEvent(
        modes=int(params['modes']), width=int(params['width']), num_static_properties=num_static,
        input_channels=dynamic_channels + 1, include_precipitation=True,
        dropout_static=float(params['dropout_static']), dropout_fc=float(params['dropout_fc']),
    )


def suggest_parameters(trial):
    step = import_module(f"{ROOT.name}.02_FNO_train_optuna")
    return step.suggest_parameters(trial)


def estimation_attributes(models):
    return {'sequence_length_observations': OBSERVATION_EVENT_SEQUENCE_LENGTH}


FAMILY = ModelFamily('FNO', 'Fourier_Neural_Operator', build_model, suggest_parameters,
                     estimation_attributes=estimation_attributes, label='full')
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
    workflow = import_module('Fourier_Neural_Operator.04_FNO_train').workflow
    args = build_argument_parser().parse_args()
    selection = workflow.load_parameters()
    validate_execution(args.gpu, args.workers)
    jobs = prepare_ensemble_jobs(workflow, selection['parameters'], args.gpu)
    fits = workflow.run_jobs(jobs, args.workers, epoch_bars=[f'Seed {seed:>3}' for seed in MODEL_SEEDS])
    save_ensemble(workflow, selection, fits)
    print("04_FNO_train.py is finished.")
