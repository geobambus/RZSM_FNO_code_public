"""Atomic CSV output and narrow cleanup helpers for EF predictions."""

from __future__ import annotations

import glob
import os


def atomic_write_csv(dataframe, path):
    """Atomically write a pandas DataFrame without its index."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    temporary_path = f"{path}.tmp.{os.getpid()}"
    try:
        dataframe.to_csv(temporary_path, index=False)
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)


def clean_matching_files(directory, pattern):
    """Remove only regular files matching a narrow run-specific pattern."""
    removed = []
    for path in glob.glob(os.path.join(directory, pattern)):
        if os.path.isfile(path):
            os.remove(path)
            removed.append(path)
    return removed
