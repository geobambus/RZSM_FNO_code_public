'''Estimate CONUS RZSM from SMAP with the pixel-trained LSTM ensemble.'''

import argparse
import sys
from importlib import import_module
from pathlib import Path

PUBLICATION_ROOT = Path(__file__).resolve().parents[1]
if str(PUBLICATION_ROOT) not in sys.path:
    sys.path.insert(0, str(PUBLICATION_ROOT))

from core.project_paths import DATA_ROOT, estimation_output_root
from core.rzsm_pipeline_config import ESTIMATION_YEARS


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--gpus', nargs='+', metavar='GPU',
        help='CUDA indices (default: all visible devices up to the ensemble size); use cpu to force CPU.',
    )
    args = parser.parse_args(argv)
    if args.gpus is not None:
        if [value.lower() for value in args.gpus] == ['cpu']:
            args.gpus = []
        else:
            try:
                args.gpus = [int(value) for value in args.gpus]
            except ValueError:
                parser.error('--gpus takes CUDA indices or the single word cpu.')
            if any(gpu < 0 for gpu in args.gpus):
                parser.error('--gpus indices must be nonnegative.')
    return args


def main(argv=None):
    args = parse_args(argv)

    estimation = import_module('CONUS_RZSM_Estimation.02_EF_RZSM_estimation')

    gpus = estimation.validate_gpus(args.gpus)
    print(f'Writing LSTM estimates under {estimation_output_root(DATA_ROOT)}')
    estimation.process_conus('SMAP', ['LSTM'], gpus=gpus, workers=10)
    estimation.merge_one('LSTM', 'SMAP', ESTIMATION_YEARS)


if __name__ == '__main__':
    main()
