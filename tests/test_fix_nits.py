"""Regression tests for four nits identified in the review cycle.

Each test is written to FAIL (RED) against the current code and to PASS once
the corresponding fix lands (GREEN):

- A: ``RainfallAnalyzer.__init__`` must not expose the dead ``logger_level``
  parameter (nor store it as an instance attribute).
- B: ``HydroData(uri="")`` must raise ``ValueError``, not a cryptic
  ``FileNotFoundError`` from a bogus relative path.
- C: ``rainfall_runoff_event_identify`` / ``get_rr_events`` in
  ``dmca_esr.py`` must not print debug output to stdout.
- D: ``scripts/station_dataset_usage.py`` must not pass the legacy
  ``data_path=`` / ``dataset_name=`` keyword arguments (replaced by ``uri=``).

Only this test file is created; no source files are modified.
"""

import pytest


# ── A. RainfallAnalyzer.logger_level is a dead parameter ──


class TestRainfallAnalyzerLoggerLevel:
    """``logger_level`` on ``RainfallAnalyzer`` is never read anywhere."""

    def test_init_signature_has_no_logger_level_parameter(self):
        """RainfallAnalyzer.__init__ must not accept a logger_level argument."""
        import inspect

        from hydrodatasource.cleaner.rainfall_cleaner import RainfallAnalyzer

        parameters = inspect.signature(RainfallAnalyzer.__init__).parameters
        assert "logger_level" not in parameters

    def test_instance_has_no_logger_level_attribute(self, tmp_path):
        """A RainfallAnalyzer instance must not carry a logger_level attribute."""
        from hydrodatasource.cleaner.rainfall_cleaner import RainfallAnalyzer

        data_dir = tmp_path / "data"
        out_dir = tmp_path / "out"
        data_dir.mkdir(parents=True)
        out_dir.mkdir(parents=True)

        analyzer = RainfallAnalyzer(
            data_folder=str(data_dir),
            output_folder=str(out_dir),
        )

        assert not hasattr(analyzer, "logger_level")


# ── B. HydroData(uri="") must raise ValueError ──


class TestHydroDataRejectsEmptyUri:
    """An empty-string ``uri`` must be rejected just like ``None``."""

    def test_empty_string_uri_raises_valueerror(self):
        """SelfMadeHydroDataset(uri="") must raise ValueError."""
        from hydrodatasource.reader.data_source import SelfMadeHydroDataset

        with pytest.raises(ValueError):
            SelfMadeHydroDataset(uri="")


# ── C. dmca_esr.py must not print debug output to stdout ──


class TestDmcaEsrNoStdoutPrints:
    """The DMCA-ESR event identification functions must stay silent on stdout."""

    def test_rainfall_runoff_event_identify_has_no_print(self):
        """rainfall_runoff_event_identify body must contain no print() call."""
        import inspect

        from hydrodatasource.processor.dmca_esr import rainfall_runoff_event_identify

        source = inspect.getsource(rainfall_runoff_event_identify)
        assert "print(" not in source

    def test_get_rr_events_has_no_print(self):
        """get_rr_events body must contain no print() call."""
        import inspect

        from hydrodatasource.processor.dmca_esr import get_rr_events

        source = inspect.getsource(get_rr_events)
        assert "print(" not in source


# ── D. scripts/station_dataset_usage.py must drop legacy kwargs ──


class TestStationDatasetUsageNoLegacyKwargs:
    """The usage script must construct StationHydroDataset with ``uri=``."""

    @staticmethod
    def _script_text():
        from pathlib import Path

        script_path = (
            Path(__file__).resolve().parent.parent
            / "scripts"
            / "station_dataset_usage.py"
        )
        return script_path.read_text(encoding="utf-8")

    def test_no_legacy_data_path_keyword(self):
        """The script must not pass the legacy data_path= argument."""
        assert "data_path=" not in self._script_text()

    def test_no_legacy_dataset_name_keyword(self):
        """The script must not pass the legacy dataset_name= argument."""
        assert "dataset_name=" not in self._script_text()
