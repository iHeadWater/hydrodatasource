"""Guard tests: previously-identified dead code must stay deleted.

Purpose
-------
A prior review pass flagged a batch of dead-code candidates (functions/modules
with no callers anywhere in the repository). The implementer deletes them. These
tests guard the invariant "the dead symbols no longer appear in the source tree".

These are source-inspection tests (same style as ``test_fix_pint_imports.py``):
they read the relevant ``.py`` file and assert the symbol's definition line is
absent. They deliberately avoid marking ``internal_data`` and never import the
package at module scope, so collection is side-effect free.

Candidate inventory
-------------------
A. ``hydrodatasource/utils/utils.py`` dead helper functions (superseded by the
   hydroutils wrapper ``streamflow_unit_conv``, which imports its helpers from
   ``hydroutils.hydro_units``, not from the local module):
     creatspinc, regen_box, cf2datetime, generate_time_intervals,
     _convert_target_unit, _process_custom_unit, _get_unit_conversion_info,
     _get_actual_source_unit, _normalize_unit, _is_inverse_conversion,
     _validate_inverse_consistency
   The last three only reference each other; none is referenced by the active
   functions ``streamflow_unit_conv`` / ``minio_file_list`` / ``is_minio_folder`` /
   ``calculate_basin_offsets`` / ``cal_area_from_shp``.

B. ``hydrodatasource/reader/minio_api.py``: whole module, never imported.

C. ``hydrodatasource/cleaner/rsvr_inflow_cleaner.py``:
   ``_rsvr_rolling_window_abrupt_abnormal_rm``, never called.

D. ``hydrodatasource/processor/basin_mean_rainfall.py``:
   ``calculate_voronoi_polygons``, ``plot_voronoi_polygons``,
   ``_plot_voronoi_polygons`` (the active functions ``calculate_weighted_rainfall``
   and ``calculate_thiesen_polygons`` are untouched).
"""

from pathlib import Path

# Paths are resolved at import time from this file's location; this does not
# import the package, so it is collection-safe.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_UTILS_PATH = _REPO_ROOT / "hydrodatasource" / "utils" / "utils.py"
_MINIO_API_PATH = _REPO_ROOT / "hydrodatasource" / "reader" / "minio_api.py"
_RSVR_CLEANER_PATH = (
    _REPO_ROOT / "hydrodatasource" / "cleaner" / "rsvr_inflow_cleaner.py"
)
_BASIN_MEAN_PATH = (
    _REPO_ROOT / "hydrodatasource" / "processor" / "basin_mean_rainfall.py"
)


def _source(path: Path) -> str:
    """Return the current UTF-8 text of a source file."""
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# A. hydrodatasource/utils/utils.py — dead helper functions
# ---------------------------------------------------------------------------


def test_utils_creatspinc_removed():
    """The dead netCDF grid-writing helper ``creatspinc`` must be gone."""
    assert "def creatspinc" not in _source(_UTILS_PATH), (
        "hydrodatasource/utils/utils.py still defines creatspinc, which has no "
        "callers anywhere in the repository."
    )


def test_utils_regen_box_removed():
    """The dead bbox-realignment helper ``regen_box`` must be gone."""
    assert "def regen_box" not in _source(_UTILS_PATH), (
        "hydrodatasource/utils/utils.py still defines regen_box, which has no "
        "callers anywhere in the repository."
    )


def test_utils_cf2datetime_removed():
    """The dead CF-time-coordinate helper ``cf2datetime`` must be gone."""
    assert "def cf2datetime" not in _source(_UTILS_PATH), (
        "hydrodatasource/utils/utils.py still defines cf2datetime, which has no "
        "callers anywhere in the repository."
    )


def test_utils_generate_time_intervals_removed():
    """The dead time-interval generator ``generate_time_intervals`` must be gone."""
    assert "def generate_time_intervals" not in _source(_UTILS_PATH), (
        "hydrodatasource/utils/utils.py still defines generate_time_intervals, "
        "which has no callers anywhere in the repository."
    )


def test_utils__convert_target_unit_removed():
    """The dead unit-parsing helper ``_convert_target_unit`` must be gone."""
    assert "def _convert_target_unit" not in _source(_UTILS_PATH), (
        "hydrodatasource/utils/utils.py still defines _convert_target_unit, which "
        "has no callers anywhere in the repository."
    )


def test_utils__process_custom_unit_removed():
    """The dead custom-unit helper ``_process_custom_unit`` must be gone."""
    assert "def _process_custom_unit" not in _source(_UTILS_PATH), (
        "hydrodatasource/utils/utils.py still defines _process_custom_unit, which "
        "has no callers anywhere in the repository."
    )


def test_utils__get_unit_conversion_info_removed():
    """The dead conversion-info helper ``_get_unit_conversion_info`` must be gone."""
    assert "def _get_unit_conversion_info" not in _source(_UTILS_PATH), (
        "hydrodatasource/utils/utils.py still defines _get_unit_conversion_info, "
        "which has no callers anywhere in the repository."
    )


def test_utils__get_actual_source_unit_removed():
    """The dead source-unit helper ``_get_actual_source_unit`` must be gone."""
    assert "def _get_actual_source_unit" not in _source(_UTILS_PATH), (
        "hydrodatasource/utils/utils.py still defines _get_actual_source_unit, "
        "which has no callers anywhere in the repository."
    )


def test_utils__normalize_unit_removed():
    """The dead unit-normalization helper ``_normalize_unit`` must be gone."""
    assert "def _normalize_unit" not in _source(_UTILS_PATH), (
        "hydrodatasource/utils/utils.py still defines _normalize_unit, which is "
        "only referenced by the dead _is_inverse_conversion helper."
    )


def test_utils__is_inverse_conversion_removed():
    """The dead inverse-direction helper ``_is_inverse_conversion`` must be gone."""
    assert "def _is_inverse_conversion" not in _source(_UTILS_PATH), (
        "hydrodatasource/utils/utils.py still defines _is_inverse_conversion, "
        "which is only referenced by the dead _validate_inverse_consistency helper."
    )


def test_utils__validate_inverse_consistency_removed():
    """The local dead helper ``_validate_inverse_consistency`` must be gone.

    ``streamflow_unit_conv`` imports its own copy from
    ``hydroutils.hydro_units``, so the module-level def has no consumers.
    """
    assert "def _validate_inverse_consistency" not in _source(_UTILS_PATH), (
        "hydrodatasource/utils/utils.py still defines the local "
        "_validate_inverse_consistency, which is never called; the active "
        "streamflow_unit_conv uses hydroutils.hydro_units._validate_inverse_consistency."
    )


# ---------------------------------------------------------------------------
# B. hydrodatasource/reader/minio_api.py — whole module
# ---------------------------------------------------------------------------


def test_minio_api_module_removed():
    """The reader/minio_api module (no importers anywhere) must be deleted."""
    assert not _MINIO_API_PATH.exists(), (
        "hydrodatasource/reader/minio_api.py still exists, but no file in the "
        "repository imports it."
    )


# ---------------------------------------------------------------------------
# C. hydrodatasource/cleaner/rsvr_inflow_cleaner.py
# ---------------------------------------------------------------------------


def test_rsvr_inflow_cleaner__rsvr_rolling_window_abrupt_abnormal_rm_removed():
    """The never-called rolling-window method must be gone."""
    assert "def _rsvr_rolling_window_abrupt_abnormal_rm" not in _source(
        _RSVR_CLEANER_PATH
    ), (
        "hydrodatasource/cleaner/rsvr_inflow_cleaner.py still defines "
        "_rsvr_rolling_window_abrupt_abnormal_rm, which is never called."
    )


# ---------------------------------------------------------------------------
# D. hydrodatasource/processor/basin_mean_rainfall.py — voronoi helpers
# ---------------------------------------------------------------------------


def test_basin_mean_rainfall_calculate_voronoi_polygons_removed():
    """The deprecated ``calculate_voronoi_polygons`` must be gone."""
    assert "def calculate_voronoi_polygons" not in _source(_BASIN_MEAN_PATH), (
        "hydrodatasource/processor/basin_mean_rainfall.py still defines "
        "calculate_voronoi_polygons, which has no callers anywhere in the "
        "repository."
    )


def test_basin_mean_rainfall_plot_voronoi_polygons_removed():
    """The deprecated ``plot_voronoi_polygons`` must be gone."""
    assert "def plot_voronoi_polygons" not in _source(_BASIN_MEAN_PATH), (
        "hydrodatasource/processor/basin_mean_rainfall.py still defines "
        "plot_voronoi_polygons, which has no callers anywhere in the repository."
    )


def test_basin_mean_rainfall__plot_voronoi_polygons_removed():
    """The dead private ``_plot_voronoi_polygons`` must be gone."""
    assert "def _plot_voronoi_polygons" not in _source(_BASIN_MEAN_PATH), (
        "hydrodatasource/processor/basin_mean_rainfall.py still defines "
        "_plot_voronoi_polygons, which is only referenced by the dead "
        "plot_voronoi_polygons wrapper."
    )
