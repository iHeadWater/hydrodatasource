"""Regression tests: reader modules must not import pint_xarray as a side effect.

Root cause
----------
`hydrodatasource/reader/data_source.py` (lines ~510, ~2786) and
`hydrodatasource/reader/gages.py` (line ~657) contained dead, function-level
`import pint_xarray  # noqa: F401` statements that served no purpose in the
function bodies.

pint-xarray performs a *global* side effect on import: it registers the
``.pint`` xarray accessor on every xarray object (pint_xarray/accessors.py,
module level). Under a full-suite run, tests that execute those functions
first (e.g. ``test_tg_hydro_datasource.py``) caused ``pint_xarray`` to be
imported, which made hydroutils 0.2.0's ``_convert_xarray`` take the
``hasattr(data[key], "pint")`` branch and fail on ``mm/3h`` units with
"ValueError: Unit expression cannot have a scaling factor."

The fix is to delete those dead import lines. These tests guard the invariant
"importing our reader modules and running the attribute-caching entry points
must not import pint_xarray (and thereby register the global .pint accessor)".

These tests deliberately avoid marking ``internal_data`` and keep every package
import inside a function body so collection is side-effect free.
"""

import re
import subprocess
import sys
from pathlib import Path

# Paths are resolved at import time from this file's location; this does not
# import the package, so it is collection-safe.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_DATA_SOURCE_PATH = _REPO_ROOT / "hydrodatasource" / "reader" / "data_source.py"
_GAGES_PATH = _REPO_ROOT / "hydrodatasource" / "reader" / "gages.py"

# Matches a line that imports pint_xarray (either form), ignoring indentation.
_IMPORT_RE = re.compile(
    r"^\s*(?:import\s+pint_xarray\b|from\s+pint_xarray\b)", re.MULTILINE
)

# Run in a fresh subprocess so sys.modules starts clean and the global side
# effect of importing pint_xarray is observable.
_SUBPROCESS_SNIPPET = r"""
import sys

import hydrodatasource.reader.data_source as ds
import hydrodatasource.reader.gages as gages

# Baseline: merely importing the reader modules must not pull pint_xarray in.
if "pint_xarray" in sys.modules:
    print("BASELINE: pint_xarray already imported by a reader module")
    sys.exit(1)


def _run(entry_point, cls):
    # Call an attribute-caching entry point on a bare instance so it executes
    # through its first statements (where the dead `import pint_xarray` lived),
    # then swallow the AttributeError raised because __init__ setup is absent.
    try:
        entry_point.__get__(object.__new__(cls))()
    except Exception:
        pass


_run(ds.SelfMadeHydroDataset.cache_attributes_xrdataset, ds.SelfMadeHydroDataset)
_run(ds.TgHydroDatasource.cache_intermediate_attributes_xrdataset, ds.TgHydroDatasource)
_run(gages.Gages.cache_attributes_xrdataset, gages.Gages)

if "pint_xarray" in sys.modules:
    print("SIDE EFFECT: pint_xarray imported by attribute-caching entry point")
    sys.exit(2)

print("CLEAN")
sys.exit(0)
"""


def test_data_source_has_no_pint_xarray_import():
    """The data_source reader must not contain any pint_xarray import statement."""
    source = _DATA_SOURCE_PATH.read_text(encoding="utf-8")
    assert not _IMPORT_RE.search(source), (
        "hydrodatasource/reader/data_source.py still contains an "
        "`import pint_xarray` statement; importing it registers the global "
        ".pint xarray accessor and breaks hydroutils unit conversion under a "
        "full-suite run."
    )


def test_gages_has_no_pint_xarray_import():
    """The gages reader must not contain any pint_xarray import statement."""
    source = _GAGES_PATH.read_text(encoding="utf-8")
    assert not _IMPORT_RE.search(source), (
        "hydrodatasource/reader/gages.py still contains an "
        "`import pint_xarray` statement; importing it registers the global "
        ".pint xarray accessor and breaks hydroutils unit conversion under a "
        "full-suite run."
    )


def test_reader_attribute_caching_does_not_import_pint_xarray_side_effect():
    """Running the attribute-caching entry points must not import pint_xarray.

    In a fresh subprocess: import the reader modules, then execute the three
    attribute-caching methods that historically contained a dead
    ``import pint_xarray``. If any of them imports pint_xarray, the process
    exits non-zero (returncode 2), failing this test.
    """
    proc = subprocess.run(
        [sys.executable, "-c", _SUBPROCESS_SNIPPET],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, (
        "Importing hydrodatasource.reader.data_source / gages and running the "
        "attribute-caching entry points imported pint_xarray as a global side "
        "effect (which registers the .pint xarray accessor process-wide). "
        f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )
