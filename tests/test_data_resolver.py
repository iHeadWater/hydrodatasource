"""Tests for hydrodatasource.configs.data_resolver."""

import sys
import tempfile
import types
from pathlib import Path

import pytest
import yaml

from hydrodataset.configs.data_resolver import ResolverContext

import hydrodatasource.reader.floodevent as _real_floodevent_module


class TestReaderAliases:
    """Verify READER_ALIASES covers all hydrodatasource reader classes."""

    def test_reader_aliases_not_empty(self):
        from hydrodatasource.configs.data_resolver import READER_ALIASES

        assert isinstance(READER_ALIASES, dict)
        assert len(READER_ALIASES) >= 11

    def test_floodevent_alias_registered(self):
        from hydrodatasource.configs.data_resolver import READER_ALIASES

        assert "floodevent" in READER_ALIASES
        spec = READER_ALIASES["floodevent"]
        assert spec["module"] == "hydrodatasource.reader.floodevent"
        assert spec["class"] == "FloodEventDatasource"
        assert spec["category"] == "hydrodatasource"

    def test_selfmade_alias_registered(self):
        from hydrodatasource.configs.data_resolver import READER_ALIASES

        assert "selfmade" in READER_ALIASES
        spec = READER_ALIASES["selfmade"]
        assert spec["class"] == "SelfMadeHydroDataset"

    @pytest.mark.parametrize(
        "alias",
        [
            "floodevent",
            "selfmade",
            "longterm",
            "forecast",
            "station",
            "tghydro",
            "gages",
            "grdc",
            "rainfall",
            "crd",
            "rsvrinflow",
        ],
    )
    def test_all_aliases_have_required_fields(self, alias):
        from hydrodatasource.configs.data_resolver import READER_ALIASES

        spec = READER_ALIASES[alias]
        assert "module" in spec, f"{alias} missing 'module'"
        assert "class" in spec, f"{alias} missing 'class'"
        assert "category" in spec, f"{alias} missing 'category'"
        assert spec["category"] == "hydrodatasource"


class TestDatasetResolutionError:
    """Verify DatasetResolutionError is importable."""

    def test_error_importable(self):
        from hydrodatasource.configs.data_resolver import DatasetResolutionError

        err = DatasetResolutionError("test message")
        assert isinstance(err, ValueError)
        assert str(err) == "test message"


class TestResolveDataPath:
    """Verify resolve_data_path with mocked storage and registry."""

    @pytest.fixture
    def temp_root(self, tmp_path):
        """Create a temp directory that serves as storage.local.root."""
        data_dir = tmp_path / "data"
        data_dir.mkdir(parents=True)
        dataset_dir = data_dir / "projects" / "songliao" / "event"
        dataset_dir.mkdir(parents=True)
        return tmp_path

    @pytest.fixture
    def datasets_yml(self, tmp_path):
        """Create a minimal configs/datasets.yml."""
        configs_dir = tmp_path / "configs"
        configs_dir.mkdir(parents=True)
        yml_path = configs_dir / "datasets.yml"
        yml_path.write_text(
            yaml.dump(
                {
                    "datasets": {
                        "songliao_event": {
                            "reader": "floodevent",
                            "path": "projects/songliao/event",
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        return tmp_path

    def test_resolve_local_path(self, temp_root, datasets_yml):
        """Resolve a dataset to an absolute local path."""
        from hydrodatasource.configs.data_resolver import resolve_data_path

        ctx = ResolverContext(
            storage={"local": {"root": str(temp_root / "data")}},
            project_root=datasets_yml,
        )

        uri = resolve_data_path("songliao_event", source="local", ctx=ctx)
        expected = str(temp_root / "data" / "projects" / "songliao" / "event")
        assert uri == expected

    def test_resolve_unknown_dataset_raises(self, datasets_yml):
        """Resolving an unknown dataset id should raise."""
        from hydrodatasource.configs.data_resolver import (
            resolve_data_path,
            DatasetResolutionError,
        )

        ctx = ResolverContext(project_root=datasets_yml)

        with pytest.raises(DatasetResolutionError):
            resolve_data_path("nonexistent_dataset", source="local", ctx=ctx)

    def test_invalid_source_raises(self, datasets_yml, tmp_path):
        """Invalid source should raise."""
        from hydrodatasource.configs.data_resolver import (
            resolve_data_path,
            DatasetResolutionError,
        )

        ctx = ResolverContext(
            storage={"local": {"root": str(tmp_path / "data")}},
            project_root=datasets_yml,
        )

        with pytest.raises(DatasetResolutionError):
            resolve_data_path("songliao_event", source="ftp", ctx=ctx)

    def test_resolve_cloud_s3_uri(self, datasets_yml):
        """Resolve a dataset to an S3 URI for cloud source."""
        from hydrodatasource.configs.data_resolver import resolve_data_path

        ctx = ResolverContext(
            storage={"s3": {"bucket": "hydro-data", "prefix": "hydromodel"}},
            project_root=datasets_yml,
        )

        uri = resolve_data_path("songliao_event", source="cloud", ctx=ctx)
        assert uri == "s3://hydro-data/hydromodel/projects/songliao/event"

    def test_resolve_with_local_root_override(self, tmp_path):
        """local_root overrides storage.local.root for project-level override."""
        from hydrodatasource.configs.data_resolver import resolve_data_path

        custom_root = tmp_path / "custom"
        custom_root.mkdir(parents=True)
        dataset_dir = custom_root / "projects" / "songliao" / "event"
        dataset_dir.mkdir(parents=True)

        ctx = ResolverContext(
            storage={"local": {"root": str(custom_root)}},
        )

        uri = resolve_data_path("songliao_event", source="local", ctx=ctx)
        expected = str(dataset_dir)
        assert uri == expected

    def test_resolves_via_injected_registry_without_yaml(self, tmp_path):
        """songliao_event resolves via HDS_DATASETS injection (no YAML needed).

        The in-code HDS_DATASETS registry is the library's source of truth.
        YAML files (configs/datasets.yml) are for user projects — the library
        does not ship one. Resolution must work without any YAML file present.
        """
        from hydrodatasource.configs.data_resolver import resolve_data_path

        # Create temp data directory (no YAML anywhere)
        data_dir = tmp_path / "data"
        data_dir.mkdir(parents=True)
        dataset_dir = data_dir / "projects" / "songliao" / "event"
        dataset_dir.mkdir(parents=True)

        ctx = ResolverContext(
            storage={"local": {"root": str(data_dir)}},
            project_root=tmp_path,
        )

        # Pass a project_root that has NO configs/datasets.yml
        # Resolution must succeed via HDS_DATASETS injection alone
        uri = resolve_data_path("songliao_event", source="local", ctx=ctx)
        expected = str(dataset_dir)
        assert uri == expected

    def test_extra_registry_contains_songliao_event(self):
        """HDS_DATASETS hardcoded registry contains the songliao_event entry."""
        from hydrodatasource.configs.data_resolver import HDS_DATASETS

        assert "songliao_event" in HDS_DATASETS
        spec = HDS_DATASETS["songliao_event"]
        assert spec["reader"] == "floodevent"
        assert spec["path"] == "projects/songliao/event"


class TestWithHdsExtras:
    """Verify _with_hds_extras injects HDS datasets and aliases correctly."""

    def test_none_ctx_returns_hds_extras(self):
        """None input returns context with HDS registry and aliases."""
        from hydrodatasource.configs.data_resolver import (
            HDS_DATASETS,
            _HDS_READER_ALIASES,
            _with_hds_extras,
        )

        ctx = _with_hds_extras(None)
        assert HDS_DATASETS in ctx.extra_registry_dicts
        assert ctx.extra_reader_aliases == _HDS_READER_ALIASES

    def test_provided_ctx_is_merged(self):
        """Caller extras are preserved; HDS extras are appended on top."""
        from hydrodatasource.configs.data_resolver import (
            HDS_DATASETS,
            _HDS_READER_ALIASES,
            _with_hds_extras,
        )

        caller_dict = {"my_ds": {"reader": "my_reader", "path": "my/path"}}
        caller_alias = {"my_reader": {"module": "m", "class": "C", "category": "t"}}
        base_ctx = ResolverContext(
            extra_registry_dicts=[caller_dict],
            extra_reader_aliases=caller_alias,
        )

        merged = _with_hds_extras(base_ctx)
        # Caller dict is first; HDS dict is last (override).
        assert caller_dict in merged.extra_registry_dicts
        assert HDS_DATASETS in merged.extra_registry_dicts
        assert merged.extra_registry_dicts.index(
            HDS_DATASETS
        ) > merged.extra_registry_dicts.index(caller_dict)
        # HDS aliases override caller aliases on conflict.
        for key, spec in _HDS_READER_ALIASES.items():
            assert merged.extra_reader_aliases[key] == spec


class TestOpenDataset:
    """Tests for hydrodatasource.configs.data_resolver.open_dataset."""

    @pytest.fixture
    def fake_flood_ctx(self, tmp_path):
        """Patch FloodEventDatasource so open_dataset can be tested without real data."""
        dataset_dir = tmp_path / "projects" / "songliao" / "event"
        dataset_dir.mkdir(parents=True)

        constructed = {}

        class FakeFloodEvent:
            def __init__(self, uri, **kwargs):
                constructed["uri"] = uri
                constructed["kwargs"] = kwargs

        mod = types.ModuleType("hydrodatasource.reader.floodevent")
        mod.FloodEventDatasource = FakeFloodEvent
        sys.modules["hydrodatasource.reader.floodevent"] = mod

        try:
            ctx = ResolverContext(
                storage={"local": {"root": str(tmp_path)}},
            )
            yield ctx, FakeFloodEvent, constructed, dataset_dir
        finally:
            sys.modules["hydrodatasource.reader.floodevent"] = _real_floodevent_module

    def test_open_dataset_songliao_event(self, fake_flood_ctx):
        """open_dataset('songliao_event') resolves reader via registry, not alias key.

        This is the core regression test: 'songliao_event' -> reader 'floodevent'
        (dataset_id != reader alias), which would KeyError without the registry lookup.
        """
        from hydrodatasource.configs.data_resolver import open_dataset

        ctx, FakeFloodEvent, constructed, dataset_dir = fake_flood_ctx
        result = open_dataset("songliao_event", ctx=ctx)

        assert isinstance(result, FakeFloodEvent)
        assert constructed["uri"] == str(dataset_dir)

    def test_open_dataset_unknown_id_raises(self, tmp_path):
        """Unknown dataset id raises DatasetResolutionError."""
        from hydrodatasource.configs.data_resolver import (
            DatasetResolutionError,
            open_dataset,
        )

        ctx = ResolverContext(storage={"local": {"root": str(tmp_path)}})
        with pytest.raises(DatasetResolutionError):
            open_dataset("nonexistent_xyz_hds", ctx=ctx)

    def test_open_dataset_exported_from_package(self):
        """open_dataset is importable from the top-level hydrodatasource package."""
        from hydrodatasource import open_dataset as pkg_open
        from hydrodatasource.configs.data_resolver import open_dataset

        assert pkg_open is open_dataset

    def test_open_dataset_forwards_kwargs(self, tmp_path):
        """Extra kwargs are forwarded to the reader constructor."""
        dataset_dir = tmp_path / "my_dataset"
        dataset_dir.mkdir()

        received = {}

        class KwargReader:
            def __init__(self, uri, time_unit=None, **kwargs):
                received["time_unit"] = time_unit

        mod = types.ModuleType("_hds_kwarg_reader_mod")
        mod.KwargReader = KwargReader
        sys.modules["_hds_kwarg_reader_mod"] = mod

        from hydrodatasource.configs.data_resolver import open_dataset

        ctx = ResolverContext(
            storage={"local": {"root": str(tmp_path)}},
            extra_registry_dicts=[
                {"kwarg_hds_ds": {"reader": "kwarg_hds_reader", "path": "my_dataset"}}
            ],
            extra_reader_aliases={
                "kwarg_hds_reader": {
                    "module": "_hds_kwarg_reader_mod",
                    "class": "KwargReader",
                    "category": "test",
                }
            },
        )

        open_dataset("kwarg_hds_ds", ctx=ctx, time_unit=["1D"])
        assert received["time_unit"] == ["1D"]

        del sys.modules["_hds_kwarg_reader_mod"]

    def test_open_dataset_cloud_s3_uri(self, tmp_path):
        """open_dataset passes an S3 URI to the reader for cloud source."""
        received = {}

        class S3Reader:
            def __init__(self, uri, **kwargs):
                received["uri"] = uri

        mod = types.ModuleType("_hds_s3_reader_mod")
        mod.S3Reader = S3Reader
        sys.modules["_hds_s3_reader_mod"] = mod

        from hydrodatasource.configs.data_resolver import open_dataset

        ctx = ResolverContext(
            storage={"s3": {"bucket": "hydro-bucket", "prefix": "hds"}},
            extra_registry_dicts=[
                {"s3_hds_ds": {"reader": "s3_hds_reader", "path": "cloud/path"}}
            ],
            extra_reader_aliases={
                "s3_hds_reader": {
                    "module": "_hds_s3_reader_mod",
                    "class": "S3Reader",
                    "category": "test",
                }
            },
        )

        open_dataset("s3_hds_ds", source="cloud", ctx=ctx)
        assert received["uri"] == "s3://hydro-bucket/hds/cloud/path"

        del sys.modules["_hds_s3_reader_mod"]
