"""Portable paths and runtime settings for the public RZSM workflows.

Input data are supplied by the user. Set ``RZSM_DATA_ROOT`` to the directory
that contains the documented input layout. Generated results and figures are
repository-local by default and can be redirected with environment variables.
"""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def _path_from_environment(variable_name, default):
    return str(Path(os.environ.get(variable_name, default)).expanduser().resolve())


def _positive_integer_from_environment(variable_name, default):
    value = int(os.environ.get(variable_name, default))
    if value < 1:
        raise ValueError(f"{variable_name} must be a positive integer")
    return value


# User-supplied input root. The legacy aliases are retained because the
# notebooks import these names, but all now resolve to one documented root.
base_FP = _path_from_environment("RZSM_DATA_ROOT", PROJECT_ROOT / "Data")
cpuserver_data = _path_from_environment("RZSM_CPUSERVER_DATA", base_FP)
nas_FP = _path_from_environment("RZSM_NAS_FP", base_FP)
das_FP = _path_from_environment("RZSM_DAS_FP", base_FP)
george_FP = _path_from_environment("RZSM_GEORGE_FP", PROJECT_ROOT)

results_FP = _path_from_environment("RZSM_RESULTS_FP", PROJECT_ROOT / "Results")
figures_FP = _path_from_environment(
    "RZSM_FIGURES_FP", PROJECT_ROOT / "Outputs" / "Figures"
)

CPU_COUNT = os.cpu_count() or 1


def _workers(variable_name, default):
    return min(CPU_COUNT, _positive_integer_from_environment(variable_name, default))


EF_WORKERS = _workers("EF_WORKERS", 8)
LSTM_TRAINING_WORKERS = _workers("LSTM_TRAINING_WORKERS", 1)
LSTM_PREDICTION_WORKERS = _workers("LSTM_PREDICTION_WORKERS", 8)
FNO_TRAINING_WORKERS = _workers("FNO_TRAINING_WORKERS", 1)
FNO_PREDICTION_WORKERS = _workers("FNO_PREDICTION_WORKERS", 8)
TCA_WORKERS = _workers("TCA_WORKERS", 8)
NUMERIC_LIBRARY_THREADS = _positive_integer_from_environment(
    "RZSM_NUMERIC_LIBRARY_THREADS", 1
)


def configure_runtime():
    """Limit numerical-library threads before importing NumPy or PyTorch."""
    thread_count = str(NUMERIC_LIBRARY_THREADS)
    for variable_name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(variable_name, thread_count)
