"""Regression tests for the ``streamflow_unit`` property of ``SelfMadeHydroDataset``.

Two defects (from review of ``hydrodatasource/reader/data_source.py``):

- The ``time_unit`` whitelist accepts ``"8D"`` (constructor check) but the
  ``streamflow_unit`` ``unit_mapping`` only covers ``{"1h", "3h", "1D", "1M"}``,
  so ``SelfMadeHydroDataset(time_unit=["8D"]).streamflow_unit`` raises
  ``KeyError: '8D'``.
- ``time_unit=None`` makes ``streamflow_unit`` iterate over ``None`` and raise
  ``TypeError: 'NoneType' object is not iterable`` instead of returning an
  empty mapping (the tolerant style used elsewhere, e.g. ``or ["1D"]`` in
  ``read_ts_xrdataset``).

Each test FAILS (RED) against the current implementation and PASSES once the
property maps ``"8D"`` to ``"mm/8d"`` (the canonical string used elsewhere in
the module, e.g. in the precipitation-unit conversion) and treats
``time_unit=None`` as an empty mapping.

The tests build a minimal Caravan-style dataset directory under ``tmp_path``
so the constructor's directory inspection works; they never touch real data.
"""


def _make_dataset_dir(tmp_path):
    """Build a minimal Caravan-style dataset directory.

    Structure::

        <tmp>/selfmade_ds/
        ├── attributes/
        │   └── attributes.csv      # basin_id,area
        └── timeseries/
            ├── 1D/                 # empty time-unit dir (for _where_ts_dir)
            └── 1M/                 # empty time-unit dir (for _where_ts_dir)

    The ``timeseries`` subdirectories are required because
    ``set_data_source_describe`` lists them via ``_where_ts_dir``.
    """
    data_dir = tmp_path / "selfmade_ds"

    attrs_dir = data_dir / "attributes"
    attrs_dir.mkdir(parents=True)
    (attrs_dir / "attributes.csv").write_text(
        "basin_id,area\n01015500,12345\n01020500,67890\n",
        encoding="utf-8",
    )

    ts_dir = data_dir / "timeseries"
    (ts_dir / "1D").mkdir(parents=True)
    (ts_dir / "1M").mkdir(parents=True)

    return data_dir


class TestStreamflowUnit8D:
    """``streamflow_unit`` must map the whitelisted ``"8D"`` time unit."""

    def test_streamflow_unit_maps_8d(self, tmp_path):
        """streamflow_unit maps "8D" -> "mm/8d" (canonical unit string)."""
        from hydrodatasource.reader.data_source import SelfMadeHydroDataset

        ds = SelfMadeHydroDataset(
            uri=str(_make_dataset_dir(tmp_path)),
            time_unit=["8D"],
        )

        assert ds.streamflow_unit == {"8D": "mm/8d"}

    def test_streamflow_unit_covers_all_supported_units_including_8d(self, tmp_path):
        """streamflow_unit covers the full supported set including "8D"."""
        from hydrodatasource.reader.data_source import SelfMadeHydroDataset

        ds = SelfMadeHydroDataset(
            uri=str(_make_dataset_dir(tmp_path)),
            time_unit=["1h", "3h", "1D", "8D", "1M"],
        )

        assert ds.streamflow_unit == {
            "1h": "mm/h",
            "3h": "mm/3h",
            "1D": "mm/d",
            "8D": "mm/8d",
            "1M": "mm/M",
        }


class TestStreamflowUnitNone:
    """``streamflow_unit`` must tolerate ``time_unit=None`` by returning {}."""

    def test_streamflow_unit_empty_when_time_unit_none(self, tmp_path):
        """time_unit=None yields an empty mapping instead of raising TypeError."""
        from hydrodatasource.reader.data_source import SelfMadeHydroDataset

        ds = SelfMadeHydroDataset(
            uri=str(_make_dataset_dir(tmp_path)),
            time_unit=None,
        )

        assert ds.streamflow_unit == {}


class TestStreamflowUnitNoRegression:
    """Existing legal units keep their canonical mappings (no regression)."""

    def test_streamflow_unit_maps_legal_units(self, tmp_path):
        """1h/3h/1D/1M keep their existing mappings."""
        from hydrodatasource.reader.data_source import SelfMadeHydroDataset

        ds = SelfMadeHydroDataset(
            uri=str(_make_dataset_dir(tmp_path)),
            time_unit=["1h", "3h", "1D", "1M"],
        )

        assert ds.streamflow_unit == {
            "1h": "mm/h",
            "3h": "mm/3h",
            "1D": "mm/d",
            "1M": "mm/M",
        }
