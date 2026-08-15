"""
Regression tests for the hydroutils 0.1.0 -> 0.2.0 migration.

Background
----------
hydroutils 0.2.0 removed the `hydroutils/hydro_log.py` module (which provided
the `hydro_logger` class decorator and `HydroWarning`) and dropped the
`from .hydro_log import *` re-export in `hydroutils/__init__.py`.

As a result, the following four modules currently fail at import time
because they still import from the removed `hydro_log` module or from the
removed re-exported `hydro_logger` name:

    - hydrodatasource.reader.gages
    - hydrodatasource.reader.rainfall_reader
    - hydrodatasource.cleaner.rainfall_cleaner
    - hydrodatasource.cleaner.rsvr_inflow_cleaner

The migration replaces every use of `hydro_log` with the Python standard
`logging` module and bumps the `hydroutils` dependency floor to `>=0.2.0`.
These tests pin that behavior.

All imports of the affected modules happen *inside* the test functions so a
broken import surfaces as a per-test failure instead of a collection error.
"""

import importlib
import logging
import tomllib
from pathlib import Path

from packaging.requirements import Requirement


# --- 1. The four affected modules import cleanly -----------------------------

def test_gages_module_importable():
    assert importlib.import_module("hydrodatasource.reader.gages") is not None


def test_rainfall_reader_module_importable():
    assert (
        importlib.import_module("hydrodatasource.reader.rainfall_reader") is not None
    )


def test_rainfall_cleaner_module_importable():
    assert (
        importlib.import_module("hydrodatasource.cleaner.rainfall_cleaner") is not None
    )


def test_rsvr_inflow_cleaner_module_importable():
    assert (
        importlib.import_module("hydrodatasource.cleaner.rsvr_inflow_cleaner")
        is not None
    )


# --- 2. Cleaner/reader classes expose a stdlib logging.Logger ----------------

def test_rainfall_cleaner_class_logger():
    from hydrodatasource.cleaner.rainfall_cleaner import RainfallCleaner

    assert isinstance(RainfallCleaner.logger, logging.Logger)
    RainfallCleaner.logger.info("smoke")


def test_rainfall_analyzer_class_logger():
    from hydrodatasource.cleaner.rainfall_cleaner import RainfallAnalyzer

    assert isinstance(RainfallAnalyzer.logger, logging.Logger)
    RainfallAnalyzer.logger.info("smoke")


def test_reservoir_inflow_backtrack_class_logger():
    from hydrodatasource.cleaner.rsvr_inflow_cleaner import ReservoirInflowBacktrack

    assert isinstance(ReservoirInflowBacktrack.logger, logging.Logger)
    ReservoirInflowBacktrack.logger.info("smoke")


# --- 3. gages module-level logger --------------------------------------------

def test_gages_module_logger():
    import hydrodatasource.reader.gages as gages

    assert isinstance(gages.logger, logging.Logger)
    gages.logger.info("smoke")


# --- 4. hydroutils dependency floor bumped to 0.2.0 --------------------------

def test_pyproject_requires_hydroutils_0_2_or_newer():
    toml_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with toml_path.open("rb") as f:
        data = tomllib.load(f)

    deps = data["project"]["dependencies"]
    hydroutils_deps = [
        d for d in deps if d.strip().lower().startswith("hydroutils")
    ]
    assert len(hydroutils_deps) == 1, (
        "expected exactly one hydroutils entry in [project].dependencies"
    )

    req = Requirement(hydroutils_deps[0])
    assert str(req.specifier) == ">=0.2.0", (
        f"hydroutils lower bound should be >=0.2.0, got {req.specifier!r}"
    )
