"""Regression tests for four Suggestion-level review findings.

A — hydrodatasource/utils/utils.py ``is_minio_folder``
    wraps *every* exception in ``except Exception`` into NotImplementedError,
    including the FileNotFoundError it raises itself (missing path) and any
    underlying FS access failure. Expected: real errors propagate unchanged.

B — hydrodatasource/cleaner/streamflow_cleaner.py:443
    ``self.processed_df[methods[0]][self.origin_df["INQ"].isna()] = np.nan``
    is a chained assignment that fires pandas SettingWithCopyWarning (and the
    pandas 3.0 ``ChainedAssignmentError`` FutureWarning). Expected: use
    ``.loc`` so no such warning is emitted.

C — hydrodatasource/reader/grdc.py
    1. the ``if __name__ == "__main__":`` block calls ``cache_grdc_daily()``,
       so simply running ``python grdc.py`` kicks off a full cache build.
       Expected: the block no longer triggers a full build.
    2. ``read_site_info()`` validates grdc_no ordering with ``assert``, which
       is stripped under ``python -O``. Expected: explicit ``raise ValueError``.

D — tests/test_hydroutils_compat.py
    ``assert len(hydroutils_deps) == 1`` falsely fails if pyproject ever adds
    hydroutils[extra] or a platform marker. Relaxed to ``assert hydroutils_deps``
    (>=1) in a separate change; this file does not repeat the fragile pattern.

All tests are RED on the current code and are the regression guard for the
implementer's fix. Imports live inside the test functions so a broken module
surfaces as a per-test failure rather than a collection error.
"""


# ── A. is_minio_folder must propagate real errors (not NotImplementedError) ──


def test_is_minio_folder_propagates_file_not_found_when_path_missing(monkeypatch):
    """A missing path (FS.exists() -> False) must raise FileNotFoundError.

    The current code catches the FileNotFoundError it raises itself and
    re-wraps it as NotImplementedError.
    """
    import pytest

    import hydrodatasource.utils.utils as utils

    class _MissingFs:
        def exists(self, url):
            return False

    monkeypatch.setattr(utils, "FS", _MissingFs())

    with pytest.raises(FileNotFoundError):
        utils.is_minio_folder("s3://bucket/missing")


def test_is_minio_folder_propagates_fs_access_error(monkeypatch):
    """An FS access failure must propagate its original exception type.

    The current code wraps the underlying RuntimeError into
    NotImplementedError, hiding the real cause from callers.
    """
    import pytest

    import hydrodatasource.utils.utils as utils

    class _RaisingFs:
        def exists(self, url):
            raise RuntimeError("s3 down")

    monkeypatch.setattr(utils, "FS", _RaisingFs())

    # NotImplementedError is a subclass of RuntimeError, so pytest.raises alone
    # would silently accept the current (wrong) NotImplementedError wrap. Pin
    # the exact type to make the test genuinely RED on the current code.
    with pytest.raises(RuntimeError) as excinfo:
        utils.is_minio_folder("s3://bucket/dir")
    assert type(excinfo.value) is RuntimeError, (
        f"FS access failure must propagate the original RuntimeError, got "
        f"{type(excinfo.value).__name__}"
    )


# ── B. anomaly_process must not use chained assignment ───────────────────────


def _write_streamflow_csv(tmp_path):
    """Write a minimal streamflow CSV (TM + INQ columns) and return its path."""
    import pandas as pd

    csv_path = tmp_path / "streamflow_input.csv"
    df = pd.DataFrame(
        {
            "TM": pd.date_range("2023-01-01", periods=48, freq="D"),
            "INQ": [100, 110, 120, 130, 95, 105, 115, 125] * 6,
        }
    )
    df.to_csv(csv_path, index=False)
    return csv_path


def test_anomaly_process_does_not_emit_setting_with_copy_warning(tmp_path):
    """anomaly_process must complete without a pandas SettingWithCopyWarning.

    Line 443 writes through a chained ``df[col][mask] = value``; the fix must
    use ``.loc``. The CSV fixture has no NaN in INQ, but the warning fires on
    the chained assignment regardless of the mask content (verified on
    pandas 2.3.3).
    """
    import warnings

    import pandas as pd

    from hydrodatasource.cleaner.streamflow_cleaner import StreamflowCleaner

    csv_path = _write_streamflow_csv(tmp_path)
    cleaner = StreamflowCleaner(data_folder=str(csv_path))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cleaner.anomaly_process(["moving_average"])

    masked_warnings = [
        w
        for w in caught
        if issubclass(w.category, pd.errors.SettingWithCopyWarning)
    ]
    assert masked_warnings == [], (
        "anomaly_process emitted SettingWithCopyWarning (chained assignment "
        "in the 'remove pre-interpolated NaNs' step): "
        f"{[str(w.message) for w in masked_warnings]}"
    )


def test_anomaly_process_does_not_use_chained_assignment_future_warning(tmp_path):
    """anomaly_process must not rely on chained assignment that pandas 3.0 breaks.

    pandas 2.3 emits a FutureWarning ("ChainedAssignmentError: behaviour will
    change in pandas 3.0!") for the same chained assignment on line 443.
    """
    import warnings

    from hydrodatasource.cleaner.streamflow_cleaner import StreamflowCleaner

    csv_path = _write_streamflow_csv(tmp_path)
    cleaner = StreamflowCleaner(data_folder=str(csv_path))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cleaner.anomaly_process(["EMA"])

    chained_assignment = [
        w
        for w in caught
        if issubclass(w.category, FutureWarning)
        and "chained assignment" in str(w.message).lower()
    ]
    assert chained_assignment == [], (
        "anomaly_process relies on chained assignment: "
        f"{[str(w.message) for w in chained_assignment]}"
    )


# ── C1. grdc.py __main__ block must not trigger a full cache build ───────────


def test_grdc_main_block_does_not_trigger_full_cache_build():
    """Running ``python hydrodatasource/reader/grdc.py`` must not build the cache.

    The current ``if __name__ == "__main__":`` block calls
    ``grdc.cache_grdc_daily()`` with no station filter, which triggers a full
    GRDC cache build on direct execution. The block must be removed or guarded
    so that only explicitly-invoked code builds the cache.
    """
    from pathlib import Path

    src_path = (
        Path(__file__).resolve().parents[1] / "hydrodatasource" / "reader" / "grdc.py"
    )
    text = src_path.read_text(encoding="utf-8")

    main_marker = 'if __name__ == "__main__":'
    main_index = text.find(main_marker)
    if main_index == -1:
        # No __main__ block at all — nothing to guard against.
        return

    main_block = text[main_index:]
    assert "cache_grdc_daily" not in main_block, (
        "the __main__ block must not call cache_grdc_daily() (full cache build)"
    )


# ── C2. read_site_info must validate with ValueError, not assert ─────────────


def test_grdc_read_site_info_raises_value_error_on_unsorted_ids(monkeypatch):
    """Unordered grdc_no must raise ValueError, not an assert (stripped under -O).

    ``read_site_info`` currently uses ``assert all(x < y ...)`` to validate
    that the shapefile's grdc_no is ascending — that check silently vanishes
    under ``python -O``. The expected behaviour is an explicit
    ``raise ValueError``.
    """
    from types import SimpleNamespace

    import pandas as pd
    import pytest

    from hydrodatasource.reader import grdc as grdc_module
    from hydrodatasource.reader.grdc import Grdc

    fake_df = pd.DataFrame(
        {"grdc_no": [5.0, 3.0, 7.0], "area": [1.0, 2.0, 3.0]}
    )
    fake_gpd = SimpleNamespace(read_file=lambda path: fake_df)
    monkeypatch.setattr(grdc_module, "gpd", fake_gpd)

    obj = Grdc.__new__(Grdc)
    obj.data_source_description = {"BASINS_SHP_FILE": "fake.shp"}

    with pytest.raises(ValueError):
        obj.read_site_info()
