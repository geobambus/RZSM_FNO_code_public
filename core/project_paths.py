"""Where the workflows read data and write results.

Set ``RZSM_DATA_ROOT`` to the folder that holds ``Results/`` and ``Figures/``.
``RZSM_RESULTS_FP`` and ``RZSM_FIGURES_FP`` override those two folders
individually. ``Results/RZSM_FNO/`` is preferred when present; otherwise
``Results/`` is used for portable and isolated test layouts. Without ``RZSM_DATA_ROOT`` the workflows use ``data/`` inside
this source tree.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[1]


def _environment_path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser()


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    results: Path
    figures: Path


def _default_results_root(root: Path) -> Path:
    """Prefer the RZSM project namespace, retaining portable legacy layouts."""
    nested = root / "Results" / "RZSM_FNO"
    return nested if nested.is_dir() else root / "Results"


def _default_figures_root(root: Path) -> Path:
    """Keep workflow figures with the relocated RZSM project."""
    nested = root / "Figures" / "RZSM_FNO"
    if nested.is_dir() or (root / "Results" / "RZSM_FNO").is_dir():
        return nested
    return root / "Figures"


def resolve_project_paths() -> ProjectPaths:
    """Return the configured roots without requiring any directory to exist."""
    root = _environment_path("RZSM_DATA_ROOT", SOURCE_ROOT / "data")
    return ProjectPaths(
        root=root,
        results=_environment_path("RZSM_RESULTS_FP", _default_results_root(root)),
        figures=_environment_path("RZSM_FIGURES_FP", _default_figures_root(root)),
    )


PATHS = resolve_project_paths()
DATA_ROOT = str(PATHS.root)
if "RZSM_DATA_ROOT" not in os.environ and not PATHS.root.exists():
    warnings.warn(
        f"RZSM_DATA_ROOT is not set, and the default data root {PATHS.root} does not exist. "
        "Set RZSM_DATA_ROOT to the folder that contains Results/ and Figures/.",
        stacklevel=2,
    )


def project_results_root(project: str | os.PathLike[str] | None = None) -> Path:
    """Results folder of ``project`` (default: the configured data root)."""
    if project is None or Path(project).expanduser().resolve() == PATHS.root.resolve():
        return PATHS.results
    return _default_results_root(Path(project).expanduser())


def ndvi_average_path(results_root=None) -> Path:
    """Prepared NDVI product shared by input preparation and CONUS estimation."""
    results = PATHS.results if results_root is None else Path(results_root)
    return results / "Normalized_Difference_Vegetation_Index" / "NDVI_averaged.nc"


def estimation_output_root(project: str | os.PathLike[str] | None = None) -> Path:
    """CONUS estimation folder shared by estimation and evaluation."""
    return project_results_root(project) / "CONUS_RZSM_Estimation" / "Estimations"


def existing_result_path(path: str | os.PathLike[str]) -> Path:
    """Resolve an existing result from either naming layout for read-only use."""
    path = Path(path)
    if path.exists():
        return path
    legacy_dirs = {
        'CONUS_RZSM_Estimation': 'CONUS_RZSM_Prediction',
        'Estimations': 'Predictions',
        'RZSM_estimation': 'RZSM_prediction',
        'RZSM_estimation_calibrated': 'RZSM_prediction_calibrated',
    }
    legacy = Path(*(legacy_dirs.get(part, part) for part in path.parts))
    if legacy.name.endswith('_estimation.nc'):
        legacy = legacy.with_name(legacy.name.removesuffix('_estimation.nc') + '_prediction.nc')
    return legacy if legacy.exists() else path


def project_auxiliary_results_root(project: str | os.PathLike[str] | None = None) -> Path:
    """Shared auxiliary products stored beside the RZSM_FNO namespace."""
    results = project_results_root(project)
    return results.parent / "Others" if results.name == "RZSM_FNO" else results


def project_figures_root(project: str | os.PathLike[str] | None = None) -> Path:
    """Workflow figure root; publication artifacts use its Figures subfolder."""
    if project is None or Path(project).expanduser().resolve() == PATHS.root.resolve():
        return PATHS.figures
    return _default_figures_root(Path(project).expanduser())
