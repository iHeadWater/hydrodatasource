"""Regression tests for three silent error-masking bugs.

Bug A — hydrodatasource/cleaner/streamflow_cleaner.py:318-323
    ``adaptive_moving_average`` wraps the window slice
    ``streamflow_data[start_date:end_date]`` in ``try/except KeyError`` whose
    handler only does ``print("WTF")``. When the slice raises a KeyError the
    helper variable ``window_data`` is never bound, so the very next line
    ``window_data.mean()`` dies with ``UnboundLocalError`` (a ``NameError``
    subclass) — the real KeyError is swallowed and masked. Debug output
    ("WTF") also leaks to stdout.

Bug B — hydrodatasource/utils/utils.py:432-453
    ``minio_file_list`` wraps ``FS.ls()`` in ``except Exception`` that prints
    and returns ``[]``, silently swallowing every S3 failure. Downstream code
    (reader/data_source.py:438) calls ``minio_file_list(ts_dir)[0]``, so any
    transient S3 error surfaces as a confusing ``IndexError: list index out
    of range`` instead of the real cause.

Bug C — hydrodatasource/utils/utils.py:456-482
    ``is_minio_folder``:
      * on an empty-but-existing directory, ``objects[0]`` raises IndexError
        which is caught and re-raised as NotImplementedError — an empty folder
        should return True; and
      * rebuilds the candidate path with ``test_object = "s3://" + objects[0]``
        even when the ``ls()`` result already carries the ``s3://`` prefix, so a
        single-file path is misreported as a folder.

These tests reproduce the bugs (RED on the current code) and act as the
regression guard for the fixes.
"""


def _keyerror_on_slice_series(data, index):
    """Build a pd.Series whose slice access raises KeyError.

    pandas >= 2.2 clamps label slices on a unique DatetimeIndex instead of
    raising, so a natural input cannot exercise the ``except KeyError`` branch
    in ``adaptive_moving_average``. This test double raises KeyError on *any*
    slice access — the exact failure the original ``except KeyError`` was
    written to guard against.
    """
    import pandas as pd

    class _KeyErrorOnSliceSeries(pd.Series):
        def __getitem__(self, key):
            if isinstance(key, slice):
                raise KeyError("simulated missing time label in slice")
            return super().__getitem__(key)

    return _KeyErrorOnSliceSeries(data, index=index)


# ── Bug A: adaptive_moving_average must not mask KeyError as NameError ─────


def test_adaptive_moving_average_surfaces_keyerror_not_nameerror_when_slice_fails():
    """A failing time-label slice must surface the real KeyError.

    The current code catches the KeyError, then hits ``window_data.mean()`` on
    the unbound ``window_data`` and raises ``UnboundLocalError`` (a NameError
    subclass). The expected behaviour is that the KeyError propagates.
    """
    import numpy as np
    import pandas as pd
    import pytest

    from hydrodatasource.cleaner.streamflow_cleaner import StreamflowCleaner

    index = pd.date_range("2020-01-01", periods=10, freq="h")
    series = _keyerror_on_slice_series(np.arange(10.0), index=index)
    cleaner = StreamflowCleaner.__new__(StreamflowCleaner)

    with pytest.raises(KeyError):
        cleaner.adaptive_moving_average(series)


def test_adaptive_moving_average_does_not_print_wtf_on_missing_slice(capsys):
    """The missing-slice path must not emit the "WTF" debug print."""
    import numpy as np
    import pandas as pd
    import pytest

    from hydrodatasource.cleaner.streamflow_cleaner import StreamflowCleaner

    index = pd.date_range("2020-01-01", periods=10, freq="h")
    series = _keyerror_on_slice_series(np.arange(10.0), index=index)
    cleaner = StreamflowCleaner.__new__(StreamflowCleaner)

    with pytest.raises(Exception):
        cleaner.adaptive_moving_average(series)

    captured = capsys.readouterr()
    assert "WTF" not in captured.out


# ── Bug B: minio_file_list must not silently swallow FS.ls() errors ────────


def test_minio_file_list_propagates_ls_error_instead_of_returning_empty_list(
    monkeypatch,
):
    """An exception from ``FS.ls()`` must propagate, not become ``[]``.

    The current code catches every exception, prints it, and returns ``[]`` —
    which turns an S3 hiccup into a misleading ``IndexError`` at the call site
    (data_source.py:438 does ``minio_file_list(ts_dir)[0]``).
    """
    import pytest

    import hydrodatasource.utils.utils as utils

    class _FailingFs:
        def ls(self, url):
            raise RuntimeError("simulated s3 listing error")

    monkeypatch.setattr(utils, "FS", _FailingFs())

    with pytest.raises(RuntimeError):
        utils.minio_file_list("s3://bucket/dir")


# ── Bug C: is_minio_folder edge-case classification ────────────────────────


def test_is_minio_folder_empty_existing_directory_returns_true(monkeypatch):
    """An existing but empty directory (no trailing slash) is still a folder.

    The current code indexes ``objects[0]`` on the empty ``ls()`` result ->
    IndexError -> swallowed into NotImplementedError, instead of True.
    """
    import hydrodatasource.utils.utils as utils

    class _EmptyDirFs:
        def exists(self, url):
            return True

        def ls(self, url):
            return []

    monkeypatch.setattr(utils, "FS", _EmptyDirFs())

    assert utils.is_minio_folder("s3://bucket/emptydir") is True


def test_is_minio_folder_single_file_path_returns_false(monkeypatch):
    """A path that resolves to a single file must not be reported as a folder.

    The current code double-prefixes the first object
    (``test_object = "s3://" + objects[0]``) even though the ``ls()`` result
    already carries the ``s3://`` prefix, so the single-file path is
    misreported as a folder (returns True) instead of False.
    """
    import hydrodatasource.utils.utils as utils

    class _SingleFileFs:
        def exists(self, url):
            return True

        def ls(self, url):
            return ["s3://bucket/dir/file.csv"]

    monkeypatch.setattr(utils, "FS", _SingleFileFs())

    assert utils.is_minio_folder("s3://bucket/dir/file.csv") is False
