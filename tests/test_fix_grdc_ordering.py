"""RED regression tests for the GRDC grdc_no ordering check and is_minio_folder's ls() path.

A — hydrodatasource/reader/grdc.py:88  ``read_site_info`` validates that the
    shapefile's ``grdc_no`` column is ascending with
    ``all(x < y for x, y in zip(...))`` over the *string* forms of the IDs.
    That is a lexicographic comparison, not a numeric one, so a
    variable-length ID pair like ``["999", "1000"]`` (numerically ascending
    but lexicographically descending because '9' > '1') is wrongly rejected,
    while ``["1000", "999"]`` (numerically descending) is wrongly accepted.
    Expected: compare the IDs as numbers.

B — hydrodatasource/utils/utils.py ``is_minio_folder`` must propagate an
    ``FS.ls()`` failure verbatim even when ``FS.exists()`` already returned
    True. The strict ``type(...) is RuntimeError`` assertion pins the exact
    type so a NotImplementedError wrap (a RuntimeError subclass) cannot
    silently satisfy ``pytest.raises(RuntimeError)``.

These tests reproduce the ordering bug (RED on the current code) and act as
the regression guard for the implementer's fix. Imports live inside the test
functions so a broken module surfaces as a per-test failure rather than a
collection error.
"""


def _make_grdc_site_reader(grdc_no_list, monkeypatch):
    """Return a Grdc instance ready for ``read_site_info``.

    ``gpd.read_file`` is stubbed to return a DataFrame whose ``grdc_no``
    column is ``grdc_no_list``. ``Grdc.__new__(Grdc)`` skips ``__init__``;
    ``read_site_info`` only needs ``data_source_description``.
    """
    from types import SimpleNamespace

    import pandas as pd

    from hydrodatasource.reader import grdc as grdc_module
    from hydrodatasource.reader.grdc import Grdc

    fake_df = pd.DataFrame(
        {
            "grdc_no": grdc_no_list,
            "area": [float(i) for i in range(len(grdc_no_list))],
        }
    )
    fake_gpd = SimpleNamespace(read_file=lambda path: fake_df)
    monkeypatch.setattr(grdc_module, "gpd", fake_gpd)

    obj = Grdc.__new__(Grdc)
    obj.data_source_description = {"BASINS_SHP_FILE": "fake.shp"}
    return obj


# ── A. read_site_info must validate grdc_no ordering numerically ────────────


def test_grdc_read_site_info_accepts_numeric_ordered_variable_length_ids(monkeypatch):
    """['999', '1000'] is numerically ascending and must not raise.

    The current lexicographic comparison sees '999' > '1000' and wrongly
    raises ValueError. Expected: numeric comparison accepts the pair.
    """
    _make_grdc_site_reader(["999", "1000"], monkeypatch).read_site_info()


def test_grdc_read_site_info_rejects_numeric_unsorted_variable_length_ids(monkeypatch):
    """['1000', '999'] is numerically descending and must raise ValueError.

    The current lexicographic comparison sees '1000' < '999' and wrongly
    accepts the unsorted pair. Expected: numeric comparison rejects it.
    """
    import pytest

    with pytest.raises(ValueError):
        _make_grdc_site_reader(["1000", "999"], monkeypatch).read_site_info()


def test_grdc_read_site_info_accepts_zero_padded_ordered_ids(monkeypatch):
    """['0005', '0010'] is numerically ascending and must not raise (regression).

    Fixed-width zero-padded IDs are already accepted by the lexicographic
    check; the numeric comparison must keep accepting them.
    """
    _make_grdc_site_reader(["0005", "0010"], monkeypatch).read_site_info()


# ── B. is_minio_folder must propagate FS.ls() errors verbatim ───────────────


def test_is_minio_folder_propagates_ls_error_when_exists_true(monkeypatch):
    """An FS.ls() failure after FS.exists()==True must propagate the original
    RuntimeError.

    The strict ``type(...) is RuntimeError`` pins the exact exception type so
    a NotImplementedError wrap (a RuntimeError subclass) cannot silently pass
    ``pytest.raises(RuntimeError)``.
    """
    import pytest

    import hydrodatasource.utils.utils as utils

    class _LsRaisingFs:
        def exists(self, url):
            return True

        def ls(self, url):
            raise RuntimeError("simulated s3 listing error")

    monkeypatch.setattr(utils, "FS", _LsRaisingFs())

    with pytest.raises(RuntimeError) as excinfo:
        utils.is_minio_folder("s3://bucket/dir")
    assert type(excinfo.value) is RuntimeError, (
        f"FS.ls() failure must propagate the original RuntimeError, got "
        f"{type(excinfo.value).__name__}"
    )
