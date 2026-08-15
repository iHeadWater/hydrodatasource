"""Regression tests for the S3 branch of
``hydrodatasource.reader.data_source.SelfMadeHydroDataset._where_attr_file``.

Defect (from review): the S3 branch returned ``attributes.csv`` unconditionally,
so an S3 dataset that stores its attributes as NetCDF could never be located.
The local branch already falls back to a ``*.nc`` file when ``attributes.csv``
is absent; the S3 branch must mirror that fallback using ``conf.FS``
(``exists`` / ``glob``) instead of ``os.path.exists`` / ``glob.glob``.

Expected fixed behavior for ``"s3://" in attr_dir``::

    if conf.FS.exists(attr_csv):
        return attr_csv
    nc_files = sorted(conf.FS.glob(os.path.join(attr_dir, "*.nc")))
    return nc_files[0] if nc_files else attr_csv

RED on current code: ``test_s3_falls_back_to_first_nc_when_no_csv`` (case 2).
The other two cases are current-behavior guards that stay GREEN.

The method uses no instance state, so each test calls it on an uninitialized
instance obtained via ``SelfMadeHydroDataset.__new__(...)``. ``conf.FS`` is
patched with a tiny inline fake exposing only ``exists`` / ``glob``.
"""


def _make_fake_fs(exists_result, glob_result):
    """Return a minimal S3-filesystem stand-in for ``conf.FS``.

    ``exists`` answers a canned boolean; ``glob`` returns a canned list of
    matching paths (as a real ``s3fs`` ``glob`` would).
    """

    class _FakeFS:
        def exists(self, path):
            return exists_result

        def glob(self, pattern):
            return glob_result

    return _FakeFS()


def test_s3_returns_attributes_csv_when_it_exists(monkeypatch):
    """S3 with an attributes.csv present -> attributes.csv is returned."""
    import os

    from hydrodatasource.reader.data_source import SelfMadeHydroDataset

    attr_dir = "s3://bucket/attrs"
    attr_csv = os.path.join(attr_dir, "attributes.csv")

    monkeypatch.setattr(
        "hydrodatasource.configs.config.FS",
        _make_fake_fs(exists_result=True, glob_result=[]),
    )

    ds = SelfMadeHydroDataset.__new__(SelfMadeHydroDataset)
    assert ds._where_attr_file(attr_dir) == attr_csv


def test_s3_falls_back_to_first_nc_when_no_csv(monkeypatch):
    """S3 without attributes.csv but with a *.nc -> the .nc file is returned."""
    import os

    from hydrodatasource.reader.data_source import SelfMadeHydroDataset

    attr_dir = "s3://bucket/attrs"
    nc_file = "s3://bucket/attrs/attrs.nc"

    monkeypatch.setattr(
        "hydrodatasource.configs.config.FS",
        _make_fake_fs(exists_result=False, glob_result=[nc_file]),
    )

    ds = SelfMadeHydroDataset.__new__(SelfMadeHydroDataset)
    assert ds._where_attr_file(attr_dir) == nc_file


def test_s3_returns_attributes_csv_when_neither_csv_nor_nc_exists(monkeypatch):
    """S3 with neither attributes.csv nor any *.nc -> attributes.csv is returned."""
    import os

    from hydrodatasource.reader.data_source import SelfMadeHydroDataset

    attr_dir = "s3://bucket/attrs"
    attr_csv = os.path.join(attr_dir, "attributes.csv")

    monkeypatch.setattr(
        "hydrodatasource.configs.config.FS",
        _make_fake_fs(exists_result=False, glob_result=[]),
    )

    ds = SelfMadeHydroDataset.__new__(SelfMadeHydroDataset)
    assert ds._where_attr_file(attr_dir) == attr_csv
