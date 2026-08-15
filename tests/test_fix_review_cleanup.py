"""Guard tests: review findings from the unified-data-interface cleanup pass.

Purpose
-------
A code review of the ``feat/unified-data-interface`` work surfaced three
hygiene issues. These tests guard the invariant that the issues stay fixed.
They are source-inspection tests (same style as ``test_fix_dead_code.py``):
they read the relevant ``.py`` files and assert on exact strings / the real
dataset registry. They deliberately avoid marking ``internal_data`` and never
import the package at module scope, so collection is side-effect free.

Issue inventory
---------------
A. Dead ``from pathlib import Path`` imports in three reader modules. The
   symbol ``Path`` is never used in any of them (only the module docstring's
   ``FilePath:`` header contains the substring "Path"). The import should be
   deleted. Guard: the exact import statement must be absent.

B. ``hydrodatasource/configs/data_resolver.py`` ``open_dataset`` docstring
   example uses the unregistered dataset id ``selfmade_basin``. The only id
   registered by hydrodatasource itself is ``songliao_event`` (``HDS_DATASETS``);
   ``camels_us`` etc. come from hydrodataset's default registry. Calling
   ``open_dataset("selfmade_basin", ...)`` would raise ``DatasetResolutionError``.
   The example must use a registered id. Guard: every dataset id appearing in an
   ``open_dataset("...")`` docstring example is a member of the real merged
   registry, and ``selfmade_basin`` appears nowhere.

C. (not guarded here) ``hydrodatasource/configs/config.py`` docstring wording
   about lazy-loading "avoid import-time side effects". A source-inspection
   assertion on docstring prose is too brittle, so it is intentionally skipped;
   the fix is left to the implementer's judgment.
"""

from pathlib import Path

# Paths are resolved at import time from this file's location; this does not
# import the package, so it is collection-safe.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_RAINFALL_READER_PATH = (
    _REPO_ROOT / "hydrodatasource" / "reader" / "rainfall_reader.py"
)
_RESERVOIR_DATASETS_PATH = (
    _REPO_ROOT / "hydrodatasource" / "reader" / "reservoir_datasets.py"
)
_RSVR_INFLOW_READER_PATH = (
    _REPO_ROOT / "hydrodatasource" / "reader" / "rsvr_inflow_reader.py"
)
_DATA_RESOLVER_PATH = (
    _REPO_ROOT / "hydrodatasource" / "configs" / "data_resolver.py"
)


def _source(path: Path) -> str:
    """Return the current UTF-8 text of a source file."""
    return path.read_text(encoding="utf-8")


def _docstring_example_dataset_ids() -> list[str]:
    """Extract dataset ids from ``open_dataset("...")`` docstring examples.

    Only the ``open_dataset`` call in the module docstring's Examples section is
    inspected; the module-level ``open_dataset`` function is not invoked.
    """
    import re

    source = _source(_DATA_RESOLVER_PATH)
    return re.findall(r'open_dataset\("([^"]+)"', source)


# ---------------------------------------------------------------------------
# A. Dead `from pathlib import Path` imports in reader modules
# ---------------------------------------------------------------------------


def test_rainfall_reader_no_dead_pathlib_import():
    """``Path`` is unused in rainfall_reader.py, so its import must be gone."""
    assert "from pathlib import Path" not in _source(_RAINFALL_READER_PATH), (
        "hydrodatasource/reader/rainfall_reader.py still imports Path, but the "
        "symbol is never used (the only 'Path' occurrence is the module "
        "docstring's FilePath: header)."
    )


def test_reservoir_datasets_no_dead_pathlib_import():
    """``Path`` is unused in reservoir_datasets.py, so its import must be gone."""
    assert "from pathlib import Path" not in _source(_RESERVOIR_DATASETS_PATH), (
        "hydrodatasource/reader/reservoir_datasets.py still imports Path, but "
        "the symbol is never used (the only 'Path' occurrence is the module "
        "docstring's FilePath: header)."
    )


def test_rsvr_inflow_reader_no_dead_pathlib_import():
    """``Path`` is unused in rsvr_inflow_reader.py, so its import must be gone."""
    assert "from pathlib import Path" not in _source(_RSVR_INFLOW_READER_PATH), (
        "hydrodatasource/reader/rsvr_inflow_reader.py still imports Path, but "
        "the symbol is never used (the only 'Path' occurrence is the module "
        "docstring's FilePath: header)."
    )


# ---------------------------------------------------------------------------
# B. data_resolver.py docstring examples must use registered dataset ids
# ---------------------------------------------------------------------------


def test_data_resolver_docstring_examples_do_not_use_selfmade_basin():
    """The docstring must not reference the unregistered ``selfmade_basin`` id."""
    assert "selfmade_basin" not in _source(_DATA_RESOLVER_PATH), (
        "hydrodatasource/configs/data_resolver.py docstring example "
        'open_dataset("selfmade_basin", ...) uses an id that is registered '
        "neither in HDS_DATASETS nor in hydrodataset's default registry; "
        "calling it would raise DatasetResolutionError. Use a registered id "
        "such as 'songliao_event' or 'camels_us'."
    )


def test_data_resolver_docstring_examples_use_registered_dataset_ids():
    """Every ``open_dataset(\"...\")`` example id must exist in the real registry.

    The registered set is computed at runtime from hydrodataset's default
    registry plus ``HDS_DATASETS`` (hydrodatasource's own entries), so the
    check tracks registry changes instead of a hardcoded whitelist.
    """
    from hydrodataset.configs.data_resolver import _DEFAULT_REGISTRY
    from hydrodatasource.configs.data_resolver import HDS_DATASETS

    registered = set(_DEFAULT_REGISTRY) | set(HDS_DATASETS)

    example_ids = _docstring_example_dataset_ids()
    assert example_ids, (
        "hydrodatasource/configs/data_resolver.py docstring must contain "
        "open_dataset(\"<id>\") examples to inspect."
    )
    for dataset_id in example_ids:
        assert dataset_id in registered, (
            f"data_resolver.py docstring example uses unregistered dataset id "
            f"{dataset_id!r}; known registered ids include 'songliao_event' "
            "and 'camels_us'."
        )
