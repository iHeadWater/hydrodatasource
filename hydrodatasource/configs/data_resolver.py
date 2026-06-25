"""
Reader alias registry and data path resolution for hydrodatasource.

Defines READER_ALIASES for all hydrodatasource reader classes and provides
resolve_data_path() and open_dataset() following the same contract as
hydromodel's ADR 0001.

This module is a thin wrapper around hydrodataset.configs.data_resolver.
It only uses the PUBLIC API of hydrodataset (resolve_data_path, open_dataset,
READER_ALIASES, DatasetResolutionError). The hydrodatasource-specific parts are
injected via extra_registry_dicts and extra_reader_aliases parameters.

Usage:
    from hydrodatasource.configs.data_resolver import (
        READER_ALIASES,
        resolve_data_path,
        open_dataset,
        DatasetResolutionError,
    )

    uri = resolve_data_path("songliao_event")
    ds = open_dataset("songliao_event")
"""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from hydrodataset.configs.data_resolver import (
    DatasetResolutionError,
    READER_ALIASES as _HD_READER_ALIASES,
    ResolverContext,
    open_dataset as _hd_open_dataset,
    resolve_data_path as _hd_resolve_data_path,
)

__all__ = [
    "DatasetResolutionError",
    "HDS_DATASETS",
    "READER_ALIASES",
    "ResolverContext",
    "open_dataset",
    "resolve_data_path",
]

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

HDS_DATASETS: Dict[str, Dict[str, str]] = {
    "songliao_event": {
        "reader": "floodevent",
        "path": "projects/songliao/event",
    },
}


def _with_hds_extras(ctx: Optional[ResolverContext]) -> ResolverContext:
    """Return a ResolverContext with HDS datasets and reader aliases injected.

    When *ctx* is None a fresh context containing only the HDS extras is
    returned.  When *ctx* is provided, the HDS datasets and aliases are merged
    on top of any extras already present in the caller's context (HDS values
    take precedence so that hydrodatasource readers are always resolvable).

    Args:
        ctx: Caller-supplied resolver context, or None.

    Returns:
        A new ResolverContext with HDS_DATASETS and _HDS_READER_ALIASES merged
        in.
    """
    if ctx is None:
        return ResolverContext(
            extra_registry_dicts=[HDS_DATASETS],
            extra_reader_aliases=_HDS_READER_ALIASES,
        )
    # Merge caller extras (first) with HDS defaults (HDS wins on overlap).
    merged_registry_dicts = list(ctx.extra_registry_dicts or [])
    merged_registry_dicts.append(HDS_DATASETS)
    merged_reader_aliases = dict(ctx.extra_reader_aliases or {})
    merged_reader_aliases.update(_HDS_READER_ALIASES)
    return ResolverContext(
        project_root=ctx.project_root,
        storage=ctx.storage,
        registry=ctx.registry,
        extra_registry_dicts=merged_registry_dicts,
        extra_reader_aliases=merged_reader_aliases,
    )


def resolve_data_path(
    dataset_id: str,
    *,
    source: Literal["local", "cloud"] = "local",
    ctx: Optional[ResolverContext] = None,
) -> str:
    """Resolve a dataset id to an absolute data path (URI).

    Thin wrapper around hydrodataset's resolve_data_path that injects
    hydrodatasource-specific datasets (HDS_DATASETS) and reader aliases
    (_HDS_READER_ALIASES) into the resolution context.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier from the registry (e.g. 'camels_us', 'songliao_event').
    source : str
        Storage backend: 'local' or 'cloud'.
    ctx : ResolverContext, optional
        Resolution context bundling project_root, storage config, registry
        overrides, and extra aliases.  When None (default), a new context
        is created with HDS_DATASETS and _HDS_READER_ALIASES injected.
        When provided, caller's extras are merged with HDS defaults.

    Returns
    -------
    str
        Absolute URI pointing to the dataset's data directory.

    Raises
    ------
    DatasetResolutionError
        If any resolution step fails.

    Examples
    --------
    >>> # Default: zero-boilerplate, reads ~/hydro_setting.yml
    >>> uri = resolve_data_path("songliao_event")

    >>> # Custom storage root via ResolverContext
    >>> ctx = ResolverContext(storage={"local": {"root": "/custom/data"}})
    >>> uri = resolve_data_path("songliao_event", ctx=ctx)
    """
    return _hd_resolve_data_path(dataset_id, source=source, ctx=_with_hds_extras(ctx))


def open_dataset(
    dataset_id: str,
    *,
    source: Literal["local", "cloud"] = "local",
    ctx: Optional[ResolverContext] = None,
    **reader_kwargs: Any,
):
    """Resolve a dataset id and return an instantiated reader object.

    Thin wrapper around hydrodataset's open_dataset that injects
    hydrodatasource-specific datasets (HDS_DATASETS) and reader aliases
    (_HDS_READER_ALIASES) into the resolution context.  Both hydrodataset
    datasets (e.g. ``'camels_us'``) and hydrodatasource datasets
    (e.g. ``'songliao_event'``) are supported via a single call.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier from the registry (e.g. 'camels_us', 'songliao_event').
    source : str
        Storage backend: 'local' or 'cloud'.
    ctx : ResolverContext, optional
        Resolution context.  When None, HDS defaults are used automatically.
    **reader_kwargs
        Extra keyword arguments forwarded to the reader constructor
        (e.g. ``time_unit=["1D"]`` for ``SelfMadeHydroDataset``).

    Returns
    -------
    object
        An instance of the reader class registered for *dataset_id*.

    Raises
    ------
    DatasetResolutionError
        If any resolution step fails.

    Examples
    --------
    >>> ds = open_dataset("songliao_event")
    >>> ds = open_dataset("camels_us", source="cloud")
    >>> ds = open_dataset("selfmade_basin", time_unit=["1D"])
    """
    return _hd_open_dataset(
        dataset_id, source=source, ctx=_with_hds_extras(ctx), **reader_kwargs
    )
