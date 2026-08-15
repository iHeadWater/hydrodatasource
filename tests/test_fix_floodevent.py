"""Regression tests for the floodevent augmented-timeseries dimension bug (C1).

Root cause
----------
The write/merge path of ``FloodEventDatasource`` stores the site dimension under
the name ``basin``:

    create_xarray_dataset_from_augdf(...)      -> coords={"time": ..., "basin": [station_id]}
    _process_augmented_files_by_paths(...)     -> xr.concat(all_datasets, dim="basin")

but ``read_ts_xrdataset_augmented`` filters on a ``gage_id`` coordinate that is
never written:

    available_stations = [
        station for station in gage_id_lst
        if station in ds.coords.get("gage_id", [])
    ]

Because no ``gage_id`` coord exists, ``available_stations`` is always ``[]`` and
the filter is skipped — so the read returns EVERY basin in the cache file instead
of only the requested subset.

Expected post-fix behaviour
---------------------------
``read_ts_xrdataset_augmented(gage_id_lst=[...])`` must filter on the ``basin``
dimension so the returned Dataset's ``basin`` coord contains exactly the
requested stations.

These tests reproduce the bug (RED on the current code) and act as the
regression guard for the fix.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import hydrodatasource.reader.floodevent as floodevent_module


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def floodevent_datasource(tmp_path):
    """Instantiate a real FloodEventDatasource on a minimal tmp dataset.

    FloodEventDatasource.__init__ validates a local filesystem layout with
    ``required_subdirs = ["attributes", "timeseries"]``, so we create exactly
    that structure (plus a ``3h`` timeseries sub-directory which
    ``set_data_source_describe`` expects to enumerate).
    """
    data_dir = tmp_path / "floodevent_test_dataset"
    attrs_dir = data_dir / "attributes"
    ts_dir = data_dir / "timeseries" / "3h"
    attrs_dir.mkdir(parents=True)
    ts_dir.mkdir(parents=True)

    pd.DataFrame(
        {
            "basin_id": ["01015500", "01015501", "01020500"],
            "area": [100.0, 150.0, 200.0],
        }
    ).to_csv(attrs_dir / "attributes.csv", index=False)

    from hydrodatasource.reader.floodevent import FloodEventDatasource

    return FloodEventDatasource(uri=str(data_dir), time_unit=["3h"])


@pytest.fixture
def augmented_cache_dir(tmp_path, monkeypatch):
    """Redirect the floodevent module's CACHE_DIR to a temp directory.

    floodevent.py binds ``CACHE_DIR`` at import time via
    ``from hydrodatasource.configs.config import CACHE_DIR``, so the module-level
    attribute must be patched (not ``config.CACHE_DIR``).
    """
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(exist_ok=True)
    monkeypatch.setattr(floodevent_module, "CACHE_DIR", str(cache_dir))
    return cache_dir


# ── Helpers ─────────────────────────────────────────────────────────────────


def _make_augmented_dataset(basins, n_time=8):
    """Mimic the write-end output: one ``basin`` coord holding all sites.

    This matches the format produced by ``create_xarray_dataset_from_augdf``
    plus ``xr.concat(..., dim="basin")``.
    """
    time = pd.date_range("2020-01-01", periods=n_time, freq="3h")
    rng = np.random.default_rng(0)
    n_basin = len(basins)
    return xr.Dataset(
        data_vars={
            "rain": (["basin", "time"], rng.random((n_basin, n_time))),
            "inflow": (["basin", "time"], rng.random((n_basin, n_time)) * 100.0),
            "flood_event": (
                ["basin", "time"],
                np.zeros((n_basin, n_time), dtype=np.int32),
            ),
            "ES": (["basin", "time"], rng.random((n_basin, n_time))),
        },
        coords={"time": time, "basin": basins},
        attrs={"description": "Augmented hydrological time series data"},
    )


def _cache_name(datasource, time_unit, first_station, last_station):
    """Recompute the augmented cache file name using the reader's own rule."""
    return (
        f"{datasource.dataset_name}_dataaugment_"
        f"timeseries_{time_unit}_batch_{first_station}_{last_station}.nc"
    )


# ── RED tests: read side must filter on the ``basin`` dimension ─────────────


def test_read_augmented_returns_only_requested_basin(
    floodevent_datasource, augmented_cache_dir
):
    """Requesting one basin from a multi-basin cache must return only that basin.

    The cache file is found (name matches the reader's lookup for a single-station
    request) but contains an extra basin. The current code skips the filter
    (no ``gage_id`` coord) and returns both basins -> RED.
    """
    ds = _make_augmented_dataset(basins=["01015500", "01020500"])
    cache_file = augmented_cache_dir / _cache_name(
        floodevent_datasource, "3h", "01015500", "01015500"
    )
    ds.to_netcdf(cache_file)

    result = floodevent_datasource.read_ts_xrdataset_augmented(
        gage_id_lst=["01015500"],
        time_unit="3h",
    )

    assert "3h" in result, "augmented read did not return the '3h' entry"
    returned_basins = result["3h"]["basin"].values.tolist()
    assert returned_basins == ["01015500"], (
        f"expected only ['01015500'], got {returned_basins} "
        f"(filter looked at 'gage_id' instead of 'basin')"
    )


def test_read_augmented_subset_of_batch_returns_only_requested(
    floodevent_datasource, augmented_cache_dir
):
    """Requesting a sub-range of a batch file must not leak extra basins.

    A batch cache file covers a range of station IDs (first/last in the name) and
    can contain intermediate basins. Requesting two of the three basins must
    return exactly those two. The buggy filter returns all three -> RED.
    """
    ds = _make_augmented_dataset(
        basins=["01015500", "01015501", "01020500"]
    )
    cache_file = augmented_cache_dir / _cache_name(
        floodevent_datasource, "3h", "01015500", "01020500"
    )
    ds.to_netcdf(cache_file)

    result = floodevent_datasource.read_ts_xrdataset_augmented(
        gage_id_lst=["01015500", "01020500"],
        time_unit="3h",
    )

    assert "3h" in result, "augmented read did not return the '3h' entry"
    returned_basins = result["3h"]["basin"].values.tolist()
    assert returned_basins == ["01015500", "01020500"], (
        f"expected ['01015500', '01020500'], got {returned_basins} "
        f"(intermediate basin '01015501' should have been filtered out)"
    )


def test_read_augmented_filters_basin_with_t_range_and_var_lst(
    floodevent_datasource, augmented_cache_dir
):
    """The basin-filter bug persists even when t_range / var_lst are supplied."""
    ds = _make_augmented_dataset(basins=["01015500", "01020500"])
    cache_file = augmented_cache_dir / _cache_name(
        floodevent_datasource, "3h", "01015500", "01015500"
    )
    ds.to_netcdf(cache_file)

    result = floodevent_datasource.read_ts_xrdataset_augmented(
        gage_id_lst=["01015500"],
        t_range=["2020-01-01", "2020-01-02"],
        var_lst=["rain", "inflow"],
        time_unit="3h",
    )

    assert "3h" in result, "augmented read did not return the '3h' entry"
    returned_basins = result["3h"]["basin"].values.tolist()
    assert returned_basins == ["01015500"], (
        f"expected only ['01015500'], got {returned_basins}"
    )


# ── GREEN spec test: the write side uses the ``basin`` dimension ────────────


def test_create_xarray_dataset_from_augdf_uses_basin_coord(floodevent_datasource):
    """Write-end format uses the ``basin`` coord (never ``gage_id``).

    This documents the source of the mismatch: the cache files carry ``basin``,
    so the read side must filter on ``basin`` too.
    """
    n_time = 8
    df = pd.DataFrame(
        {
            "time": pd.date_range("2020-01-01", periods=n_time, freq="3h"),
            "rain": np.ones(n_time),
            "inflow": np.ones(n_time) * 10.0,
            "flood_event": np.zeros(n_time, dtype=np.int32),
            "ES": np.ones(n_time) * 0.5,
        }
    )

    ds = floodevent_datasource.create_xarray_dataset_from_augdf(
        df, station_id="01015500", time_unit="3h"
    )

    assert "basin" in ds.coords, "write side must expose a 'basin' coord"
    assert "gage_id" not in ds.coords, "write side must NOT expose 'gage_id'"
    assert ds["basin"].values.tolist() == ["01015500"]
