"""
Reader alias registry and data path resolution for hydrodatasource.

Defines READER_ALIASES for all hydrodatasource reader classes and provides
resolve_data_path() following the same contract as hydromodel's ADR 0001.

Reuses hydrodataset's storage configuration parsing (settings.py) and
validation logic, so all three repos produce the same resolved URIs.

Usage:
    from hydrodatasource.configs.data_resolver import (
        READER_ALIASES,
        resolve_data_path,
        DatasetResolutionError,
    )

    uri = resolve_data_path("songliao_event", project_root=".")
    ds = FloodEventDatasource(data_path=uri, dataset_name="myevents")
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Dict, Optional

import yaml

from hydrodataset.configs.data_resolver import (
    DatasetResolutionError,
    READER_ALIASES as _HD_READER_ALIASES,
)
from hydrodataset.configs.settings import get_local_root, get_storage_config

# ── hydrodatasource-specific reader aliases ──────────────────────────────

_HDS_READER_ALIASES: Dict[str, Dict[str, str]] = {
    "floodevent": {
        "module": "hydrodatasource.reader.floodevent",
        "class": "FloodEventDatasource",
        "category": "hydrodatasource",
    },
    "selfmade": {
        "module": "hydrodatasource.reader.data_source",
        "class": "SelfMadeHydroDataset",
        "category": "hydrodatasource",
    },
    "longterm": {
        "module": "hydrodatasource.reader.data_source",
        "class": "LongTermDataset",
        "category": "hydrodatasource",
    },
    "forecast": {
        "module": "hydrodatasource.reader.data_source",
        "class": "SelfMadeForecastDataset",
        "category": "hydrodatasource",
    },
    "station": {
        "module": "hydrodatasource.reader.data_source",
        "class": "StationHydroDataset",
        "category": "hydrodatasource",
    },
    "tghydro": {
        "module": "hydrodatasource.reader.data_source",
        "class": "TgHydroDatasource",
        "category": "hydrodatasource",
    },
    "gages": {
        "module": "hydrodatasource.reader.gages",
        "class": "Gages",
        "category": "hydrodatasource",
    },
    "grdc": {
        "module": "hydrodatasource.reader.grdc",
        "class": "Grdc",
        "category": "hydrodatasource",
    },
    "rainfall": {
        "module": "hydrodatasource.reader.rainfall_reader",
        "class": "RainfallReader",
        "category": "hydrodatasource",
    },
    "crd": {
        "module": "hydrodatasource.reader.reservoir_datasets",
        "class": "Crd",
        "category": "hydrodatasource",
    },
    "rsvrinflow": {
        "module": "hydrodatasource.reader.rsvr_inflow_reader",
        "class": "RsvrInflowReader",
        "category": "hydrodatasource",
    },
}

# Merged: hydrodataset's aliases first, then hydrodatasource overrides
READER_ALIASES: Dict[str, Dict[str, str]] = {
    **_HD_READER_ALIASES,
    **_HDS_READER_ALIASES,
}

FORBIDDEN_PATH_PATTERNS = {"://", ".."}


def _load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML file, returning {} if it does not exist."""
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    return loaded or {}


def _load_registry(project_root: Path) -> Dict[str, Dict[str, Any]]:
    """Load dataset registry, checking hydrodatasource first then hydrodataset.

    Looks in these locations:
    1. {project_root}/configs/datasets.yml (project-level, takes precedence)
    2. hydrodatasource package's own configs/datasets.yml
    3. hydrodataset/configs/datasets.yml (package fallback)

    Parameters
    ----------
    project_root : Path
        Root of the calling project.

    Returns
    -------
    dict
        Dataset registry mapping dataset_id -> {'reader': ..., 'path': ...}

    Raises
    ------
    DatasetResolutionError
        If no registry file is found.
    """
    # Project-level config (takes precedence)
    project_registry = project_root / "configs" / "datasets.yml"
    # hydrodatasource package-internal fallback
    package_registry = (
        Path(__file__).resolve().parent.parent.parent / "configs" / "datasets.yml"
    )
    # hydrodataset's own datasets.yml
    hd_registry = (
        Path(__file__).resolve().parent.parent.parent.parent
        / "hydrodataset"
        / "configs"
        / "datasets.yml"
    )

    registry_path = None
    if project_registry.exists():
        registry_path = project_registry
    elif package_registry.exists():
        registry_path = package_registry
    elif hd_registry.exists():
        registry_path = hd_registry

    if registry_path is None:
        raise DatasetResolutionError(
            "Dataset registry not found. Tried:\n"
            f"  - {project_registry}\n"
            f"  - {package_registry}\n"
            f"  - {hd_registry}\n"
            "Create configs/datasets.yml in your project."
        )

    data = _load_yaml(registry_path)
    datasets = data.get("datasets")
    if not isinstance(datasets, dict):
        raise DatasetResolutionError(
            f"Dataset registry in {registry_path} must have a 'datasets' mapping."
        )
    return datasets


def _validate_relative_path(path_value: str, dataset_id: str) -> None:
    """Ensure path is a safe relative path (no URI schemes, no .., no absolute)."""
    if not isinstance(path_value, str):
        raise DatasetResolutionError(
            f"Dataset '{dataset_id}' path must be a string, "
            f"got {type(path_value)}"
        )
    for forbidden in FORBIDDEN_PATH_PATTERNS:
        if forbidden in path_value:
            raise DatasetResolutionError(
                f"Dataset '{dataset_id}' path must be relative, "
                f"not contain '{forbidden}'"
            )
    windows_path = PureWindowsPath(path_value)
    posix_path = PurePosixPath(path_value)
    if windows_path.is_absolute() or posix_path.is_absolute():
        raise DatasetResolutionError(
            f"Dataset '{dataset_id}' path must be relative, "
            f"got absolute: '{path_value}'"
        )


def resolve_data_path(
    dataset_id: str,
    *,
    source: str = "local",
    project_root: Optional[str] = None,
) -> str:
    """Resolve a dataset id to an absolute data path (URI).

    Combines the dataset registry entry with storage configuration
    to produce a single absolute path. Follows the same contract as
    hydromodel's resolve_data_cfgs and hydrodataset's resolve_data_path.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier from the registry (e.g. 'camels_us', 'songliao_event').
    source : str
        Storage backend: 'local' or 'cloud'.
    project_root : str, optional
        Root of the calling project (for finding configs/datasets.yml).
        Defaults to current working directory.

    Returns
    -------
    str
        Absolute URI pointing to the dataset's data directory.

    Raises
    ------
    DatasetResolutionError
        If any resolution step fails.
    """
    if source not in {"local", "cloud"}:
        raise DatasetResolutionError(
            f"source must be 'local' or 'cloud', got '{source}'"
        )

    root = Path(project_root) if project_root else Path.cwd()
    registry = _load_registry(root)

    if dataset_id not in registry:
        known = ", ".join(sorted(registry))
        raise DatasetResolutionError(
            f"Unknown dataset id '{dataset_id}'. "
            f"Known datasets: {known}"
        )

    dataset_spec = registry[dataset_id]
    reader = dataset_spec.get("reader")
    if not reader:
        raise DatasetResolutionError(
            f"Dataset '{dataset_id}' must define 'reader'"
        )
    if reader not in READER_ALIASES:
        raise DatasetResolutionError(
            f"Unknown reader alias '{reader}' for dataset '{dataset_id}'"
        )

    relative_path = dataset_spec.get("path")
    if not relative_path:
        raise DatasetResolutionError(
            f"Dataset '{dataset_id}' must define 'path'"
        )
    _validate_relative_path(relative_path, dataset_id)

    if source == "local":
        root_dir = get_local_root()
        if root_dir is None:
            raise DatasetResolutionError(
                "storage.local.root is not configured. "
                "Set it in ~/hydro_setting.yml"
            )
        if not root_dir.exists():
            raise DatasetResolutionError(
                f"storage.local.root does not exist: {root_dir}"
            )
        resolved = root_dir / relative_path
        if not resolved.exists():
            raise DatasetResolutionError(
                f"Resolved dataset path does not exist: {resolved}"
            )
        return str(resolved)

    # cloud source
    storage = get_storage_config()
    s3 = storage.get("s3")
    if not isinstance(s3, dict):
        raise DatasetResolutionError(
            "storage.s3 is required for cloud source"
        )
    bucket = s3.get("bucket")
    if not bucket:
        raise DatasetResolutionError("storage.s3.bucket is required")
    prefix = str(s3.get("prefix") or "").strip("/")
    rel = relative_path.replace("\\", "/").strip("/")
    path = f"{prefix}/{rel}" if prefix else rel
    return f"s3://{bucket}/{path}"
