"""
Reader alias registry and data path resolution for hydrodatasource.

Defines READER_ALIASES for all hydrodatasource reader classes and provides
resolve_data_path() following the same contract as hydromodel's ADR 0001.

This module is a thin wrapper around hydrodataset.configs.data_resolver.
It only uses the PUBLIC API of hydrodataset (resolve_data_path, READER_ALIASES,
DatasetResolutionError). The hydrodatasource-specific parts are injected via
extra_registry_dicts and extra_reader_aliases parameters.

Usage:
    from hydrodatasource.configs.data_resolver import (
        READER_ALIASES,
        resolve_data_path,
        DatasetResolutionError,
    )

    uri = resolve_data_path("songliao_event", local_root="/data/hydro")
    ds = FloodEventDatasource(uri=uri)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from hydrodataset.configs.data_resolver import (
    DatasetResolutionError,
    READER_ALIASES as _HD_READER_ALIASES,
    resolve_data_path as _hd_resolve_data_path,
)

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

# ── Dataset registry (in-code, not YAML) ──────────────────────────────────
#
# The library's dataset registry lives here in code. This is the authoritative
# source for datasets served by this package (Layer 2 in the 3-layer cascade).
#
# YAML files (configs/datasets.yml) are a Layer 3 **user override** — external
# projects using hydrodatasource or hydrodataset can provide their own
# configs/datasets.yml to customize or extend the registry. The library itself
# does not ship a default YAML.

_HDS_DATASETS: Dict[str, Dict[str, str]] = {
    "songliao_event": {
        "reader": "floodevent",
        "path": "projects/songliao/event",
    },
}


def resolve_data_path(
    dataset_id: str,
    *,
    source: str = "local",
    project_root: Optional[str] = None,
    local_root: Optional[str] = None,
) -> str:
    """Resolve a dataset id to an absolute data path (URI).

    Thin wrapper around hydrodataset's resolve_data_path that injects
    hydrodatasource-specific datasets and reader aliases.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier from the registry (e.g. 'camels_us', 'songliao_event').
    source : str
        Storage backend: 'local' or 'cloud'.
    project_root : str, optional
        Root of the calling project (for finding configs/datasets.yml).
        Defaults to current working directory.
    local_root : str, optional
        Override storage.local.root for project-level root override.
        When provided, skips reading storage.local.root from settings.

    Returns
    -------
    str
        Absolute URI pointing to the dataset's data directory.

    Raises
    ------
    DatasetResolutionError
        If any resolution step fails.
    """
    result = _hd_resolve_data_path(
        dataset_id,
        source=source,
        project_root=project_root,
        local_root=Path(local_root) if local_root else None,
        extra_registry_dicts=[_HDS_DATASETS],
        extra_reader_aliases=_HDS_READER_ALIASES,
    )
    return str(result)
