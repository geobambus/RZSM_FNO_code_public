"""Final training settings adopted for the FNO and LSTM in the paper."""

from copy import deepcopy


_PAPER_SELECTIONS = {
    "FNO": {
        "family": "FNO",
        "architecture": "full",
        "trial": 163,
        "parameters": {
            "batch_size": 1024,
            "dropout_fc": 0.0870066460040281,
            "dropout_static": 0.155802384142592,
            "eta_min": 1e-06,
            "learning_rate": 0.0004165594513519,
            "local_kernel_size": 3,
            "modes": 7,
            "scheduler_factor": 0.5,
            "scheduler_patience": 10,
            "weight_decay": 1.8424211596791644e-05,
            "width": 128
        }
    },
    "LSTM": {
        "family": "LSTM",
        "architecture": "LSTM",
        "trial": 134,
        "parameters": {
            "batch_size": 1024,
            "dropout_fc": 0.05472255765132967,
            "dropout_static": 0.08710527400581103,
            "eta_min": 1e-06,
            "hidden_size": 128,
            "learning_rate": 0.00048134616281188513,
            "num_layers": 2,
            "scheduler_factor": 0.5,
            "scheduler_patience": 10,
            "weight_decay": 0.00022449489190446735
        }
    }
}


def paper_selection(family):
    """Return a fresh copy of the published selection for one model."""
    return deepcopy(_PAPER_SELECTIONS[family])
