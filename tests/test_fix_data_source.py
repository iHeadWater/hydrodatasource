"""Regression tests for three defects in hydrodatasource.reader.data_source.

Each test is written to FAIL (RED) against the current, buggy implementation
and to PASS once the corresponding defect is fixed (GREEN):

- Defect A: ``uri=None`` silently generates a ``"None"`` path, and the legacy
  kwargs ``data_path`` / ``dataset_name`` / ``data_folder`` are silently
  swallowed by ``**kwargs`` instead of being rejected with a ``ValueError``.
- Defect B: ``basin_id`` leading zeros are lost when reading
  ``attributes/attributes.csv`` (``pd.read_csv`` infers ``int64`` for pure
  numeric IDs, so ``"01015500"`` becomes ``"1015500"``).
- Defect C: the ``streamflow_unit`` property maps ``"1MS"`` while the
  ``time_unit`` whitelist uses ``"1M"``, so accessing ``streamflow_unit`` for
  the supported unit ``"1M"`` raises ``KeyError``.

The tests are self-contained: they build a minimal Caravan-style dataset
directory under ``tmp_path`` and never touch real data.
"""

import pytest

from hydrodatasource.reader.data_source import SelfMadeHydroDataset


@pytest.fixture
def selfmade_dataset_dir(tmp_path):
    """Build a minimal Caravan-style dataset directory.

    Structure::

        <tmp>/selfmade_ds/
        ├── attributes/
        │   └── attributes.csv      # basin_id,area  (zero-padded IDs)
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


# ── Defect A: uri=None silently generates a "None" path; legacy kwargs swallowed ──


class TestConstructorRequiresUri:
    """Construction without ``uri=`` must raise ``ValueError``.

    Current bug: ``uri=None`` is passed through to ``HydroData.__init__`` as
    ``str(None) == "None"``, and legacy kwargs (``data_path``, ``dataset_name``,
    ``data_folder``) are swallowed by ``**kwargs``. The constructor then blows
    up with a ``FileNotFoundError`` on the bogus ``"None/timeseries"`` path
    instead of a clear ``ValueError``.
    """

    @pytest.mark.parametrize(
        "kwargs",
        [
            {},  # no arguments at all
            {"data_path": "/some/data"},
            {"dataset_name": "my_dataset"},
            {"data_folder": "/some/data"},
        ],
        ids=["no-args", "data_path", "dataset_name", "data_folder"],
    )
    def test_missing_uri_raises_valueerror(self, kwargs):
        """SelfMadeHydroDataset() without uri= raises ValueError."""
        with pytest.raises(ValueError):
            SelfMadeHydroDataset(**kwargs)


# ── Defect B: basin_id leading zeros are lost when reading attributes.csv ──


class TestBasinIdLeadingZeros:
    """``basin_id`` read from attributes.csv must keep zero padding.

    Current bug: ``_read_attributes_df`` reads the CSV via
    ``access_fs.spec_path`` (``pd.read_csv`` without a ``dtype``), so the pure
    numeric IDs ``01015500`` / ``01020500`` are inferred as ``int64`` and
    ``.astype(str)`` turns them into ``"1015500"`` / ``"1020500"``.
    """

    def test_basin_id_preserves_leading_zeros(self, selfmade_dataset_dir):
        """Zero-padded basin IDs survive the read as strings."""
        ds = SelfMadeHydroDataset(
            uri=str(selfmade_dataset_dir),
            time_unit=["1D"],
        )

        assert ds.camels_sites["basin_id"].tolist() == [
            "01015500",
            "01020500",
        ]


# ── Defect C: streamflow_unit mapping is missing the "1M" key ──


class TestStreamflowUnit:
    """``streamflow_unit`` must support the ``"1M"`` time unit.

    Current bug: ``streamflow_unit`` maps the key ``"1MS"`` while the
    ``time_unit`` whitelist (and therefore ``self.time_unit``) uses ``"1M"``,
    so ``unit_mapping["1M"]`` raises ``KeyError``.
    """

    def test_streamflow_unit_1m_key(self, selfmade_dataset_dir):
        """streamflow_unit maps the supported unit "1M" -> "mm/M"."""
        ds = SelfMadeHydroDataset(
            uri=str(selfmade_dataset_dir),
            time_unit=["1M"],
        )

        assert ds.streamflow_unit == {"1M": "mm/M"}

    def test_streamflow_unit_covers_all_supported_units(self, selfmade_dataset_dir):
        """streamflow_unit covers every supported time unit (1h/3h/1D/1M)."""
        ds = SelfMadeHydroDataset(
            uri=str(selfmade_dataset_dir),
            time_unit=["1h", "3h", "1D", "1M"],
        )

        assert set(ds.streamflow_unit.keys()) == {"1h", "3h", "1D", "1M"}
        assert ds.streamflow_unit["1M"] == "mm/M"
