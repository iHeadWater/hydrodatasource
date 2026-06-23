"""Tests for hydrodatasource.configs.data_resolver."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml


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

    def test_resolve_local_path(self, temp_root, datasets_yml, monkeypatch):
        """Resolve a dataset to an absolute local path."""
        from hydrodatasource.configs.data_resolver import resolve_data_path

        monkeypatch.setattr(
            "hydrodataset.configs.data_resolver.get_local_root",
            lambda: temp_root / "data",
        )

        uri = resolve_data_path(
            "songliao_event", source="local", project_root=str(datasets_yml)
        )
        expected = str(temp_root / "data" / "projects" / "songliao" / "event")
        assert uri == expected

    def test_resolve_unknown_dataset_raises(self, datasets_yml, monkeypatch, tmp_path):
        """Resolving an unknown dataset id should raise."""
        from hydrodatasource.configs.data_resolver import (
            resolve_data_path,
            DatasetResolutionError,
        )

        monkeypatch.setattr(
            "hydrodataset.configs.data_resolver.get_local_root",
            lambda: tmp_path / "data",
        )

        with pytest.raises(DatasetResolutionError):
            resolve_data_path(
                "nonexistent_dataset", source="local", project_root=str(datasets_yml)
            )

    def test_invalid_source_raises(self, datasets_yml):
        """Invalid source should raise."""
        from hydrodatasource.configs.data_resolver import (
            resolve_data_path,
            DatasetResolutionError,
        )

        with pytest.raises(DatasetResolutionError):
            resolve_data_path(
                "songliao_event", source="ftp", project_root=str(datasets_yml)
            )

    def test_resolve_cloud_s3_uri(self, datasets_yml, monkeypatch):
        """Resolve a dataset to an S3 URI for cloud source."""
        from hydrodatasource.configs.data_resolver import resolve_data_path

        monkeypatch.setattr(
            "hydrodataset.configs.data_resolver.get_storage_config",
            lambda: {
                "s3": {
                    "bucket": "hydro-data",
                    "prefix": "hydromodel",
                }
            },
        )

        uri = resolve_data_path(
            "songliao_event", source="cloud", project_root=str(datasets_yml)
        )
        assert uri == "s3://hydro-data/hydromodel/projects/songliao/event"

    def test_resolve_with_local_root_override(self, tmp_path):
        """local_root overrides storage.local.root for project-level override."""
        from hydrodatasource.configs.data_resolver import resolve_data_path

        custom_root = tmp_path / "custom"
        custom_root.mkdir(parents=True)
        dataset_dir = custom_root / "projects" / "songliao" / "event"
        dataset_dir.mkdir(parents=True)

        uri = resolve_data_path(
            "songliao_event", source="local", local_root=str(custom_root)
        )
        expected = str(dataset_dir)
        assert uri == expected

    def test_resolves_via_injected_registry_without_yaml(self, tmp_path, monkeypatch):
        """songliao_event resolves via _HDS_DATASETS injection (no YAML needed).

        The in-code _HDS_DATASETS registry is the library's source of truth.
        YAML files (configs/datasets.yml) are for user projects — the library
        does not ship one. Resolution must work without any YAML file present.
        """
        from hydrodatasource.configs.data_resolver import resolve_data_path

        # Create temp data directory (no YAML anywhere)
        data_dir = tmp_path / "data"
        data_dir.mkdir(parents=True)
        dataset_dir = data_dir / "projects" / "songliao" / "event"
        dataset_dir.mkdir(parents=True)

        monkeypatch.setattr(
            "hydrodataset.configs.data_resolver.get_local_root",
            lambda: data_dir,
        )

        # Pass a project_root that has NO configs/datasets.yml
        # Resolution must succeed via _HDS_DATASETS injection alone
        uri = resolve_data_path(
            "songliao_event", source="local", project_root=str(tmp_path)
        )
        expected = str(dataset_dir)
        assert uri == expected

    def test_extra_registry_contains_songliao_event(self):
        """_HDS_DATASETS hardcoded registry contains the songliao_event entry."""
        from hydrodatasource.configs.data_resolver import _HDS_DATASETS

        assert "songliao_event" in _HDS_DATASETS
        spec = _HDS_DATASETS["songliao_event"]
        assert spec["reader"] == "floodevent"
        assert spec["path"] == "projects/songliao/event"
