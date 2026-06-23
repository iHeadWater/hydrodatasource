"""Tests for hydrodatasource.configs.config — new public API functions.

Tests the pure-function APIs (get_local_root, get_cache_dir, get_storage_config)
and read_setting() validation, plus backward-compatible globals.

These tests verify the changes made in feat/unified-data-interface.

Only the new storage.* config format is supported. Old local_data_path.*
and minio.* formats have been removed.
"""

import os
import tempfile
from pathlib import Path

import pytest
import yaml


# ── read_setting() tests ────────────────────────────────────────────────────


class TestReadSetting:
    """Verify read_setting() accepts the new storage.* config format only."""

    def test_new_format_storage_section(self, tmp_path):
        """New format with storage.* sections is accepted."""
        from hydrodatasource.configs.config import read_setting

        setting_path = tmp_path / "hydro_setting.yml"
        setting_path.write_text(
            yaml.dump(
                {
                    "storage": {
                        "default_source": "local",
                        "local": {"root": str(tmp_path / "data")},
                        "s3": {
                            "endpoint_url": "http://minio:9000",
                            "key": "mykey",
                            "secret": "mysecret",
                            "bucket": "my-bucket",
                            "prefix": "hydromodel",
                        },
                    }
                }
            ),
            encoding="utf-8",
        )

        result = read_setting(str(setting_path))
        assert "storage" in result
        assert result["storage"]["s3"]["bucket"] == "my-bucket"

    def test_old_format_rejected(self, tmp_path):
        """Old format (local_data_path.*) raises ValueError."""
        from hydrodatasource.configs.config import read_setting

        setting_path = tmp_path / "hydro_setting.yml"
        setting_path.write_text(
            yaml.dump(
                {
                    "local_data_path": {
                        "root": str(tmp_path / "data"),
                        "datasets-origin": str(tmp_path / "data" / "datasets-origin"),
                    }
                }
            ),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="must have 'storage'"):
            read_setting(str(setting_path))

    def test_missing_file_raises(self, tmp_path):
        """Missing setting file raises FileNotFoundError."""
        from hydrodatasource.configs.config import read_setting

        with pytest.raises(FileNotFoundError, match="not found"):
            read_setting(str(tmp_path / "nonexistent.yml"))

    def test_empty_setting_raises(self, tmp_path):
        """Empty/null setting raises ValueError."""
        from hydrodatasource.configs.config import read_setting

        setting_path = tmp_path / "hydro_setting.yml"
        setting_path.write_text("", encoding="utf-8")

        with pytest.raises(ValueError, match="empty"):
            read_setting(str(setting_path))

    def test_missing_storage_section_raises(self, tmp_path):
        """Setting without 'storage' section raises ValueError."""
        from hydrodatasource.configs.config import read_setting

        setting_path = tmp_path / "hydro_setting.yml"
        setting_path.write_text(
            yaml.dump({"some_other_key": "value"}), encoding="utf-8"
        )

        with pytest.raises(ValueError, match="must have 'storage'"):
            read_setting(str(setting_path))


# ── Pure-function API tests ─────────────────────────────────────────────────


class TestGetLocalRoot:
    """Verify get_local_root() resolves correctly."""

    def test_returns_path_from_storage_local_root(self, monkeypatch):
        """get_local_root returns storage.local.root from settings."""
        from hydrodatasource.configs import config
        from pathlib import Path

        monkeypatch.setattr(
            config, "_hd_get_local_root", lambda: Path("/storage/local/root")
        )
        monkeypatch.setattr(config, "LOCAL_ROOT", "")

        result = config.get_local_root()
        assert result == Path("/storage/local/root")

    def test_returns_none_when_nothing_configured(self, monkeypatch):
        """get_local_root returns None when nothing is configured."""
        from hydrodatasource.configs import config

        monkeypatch.setattr(config, "_hd_get_local_root", lambda: None)
        monkeypatch.setattr(config, "LOCAL_ROOT", "")

        result = config.get_local_root()
        assert result is None


class TestGetCacheDir:
    """Verify get_cache_dir() delegates correctly."""

    def test_returns_path(self, monkeypatch):
        """get_cache_dir delegates to hydrodataset."""
        from hydrodatasource.configs import config

        monkeypatch.setattr(
            config, "_hd_get_cache_dir", lambda: Path("/cache/dir")
        )

        result = config.get_cache_dir()
        assert result == Path("/cache/dir")


class TestGetStorageConfig:
    """Verify get_storage_config() returns storage config dict."""

    def test_returns_dict(self, monkeypatch):
        """get_storage_config delegates to hydrodataset."""
        from hydrodatasource.configs import config

        expected = {
            "storage": {
                "s3": {"bucket": "test-bucket", "prefix": "test-prefix"},
            }
        }
        monkeypatch.setattr(config, "_hd_get_storage_config", lambda: expected)

        result = config.get_storage_config()
        assert result == expected


# ── Backward-compatible globals ──────────────────────────────────────────────


class TestBackwardCompatGlobals:
    """Verify legacy globals are still accessible after import."""

    def test_setting_is_dict(self):
        """SETTING is a dict (may be empty if no config file)."""
        from hydrodatasource.configs.config import SETTING

        assert isinstance(SETTING, dict)

    def test_local_root_is_str(self):
        """LOCAL_ROOT is a str (may be empty)."""
        from hydrodatasource.configs.config import LOCAL_ROOT

        assert isinstance(LOCAL_ROOT, str)

    def test_cache_dir_is_str(self):
        """CACHE_DIR is a str (may be empty)."""
        from hydrodatasource.configs.config import CACHE_DIR

        assert isinstance(CACHE_DIR, str)

    def test_minio_param_is_dict(self):
        """MINIO_PARAM is a dict (may be empty)."""
        from hydrodatasource.configs.config import MINIO_PARAM

        assert isinstance(MINIO_PARAM, dict)


# ── Format bridging tests (new format only, bridge removed) ────────────────


class TestFormatBridging:
    """Verify new storage.* format initializes correctly."""

    def test_no_local_data_path_in_setting(self, monkeypatch, tmp_path):
        """SETTING should NOT contain local_data_path key (bridge removed)."""
        from hydrodatasource.configs import config

        data_root = tmp_path / "data"

        def mock_load():
            return {
                "storage": {
                    "default_source": "local",
                    "local": {"root": str(data_root)},
                }
            }

        original_load = config._load_settings_from_file
        monkeypatch.setattr(config, "_load_settings_from_file", mock_load)
        config._init_settings()

        assert "local_data_path" not in config.SETTING
        assert config.LOCAL_ROOT == str(data_root)

        # Restore
        monkeypatch.setattr(config, "_load_settings_from_file", original_load)
        config._init_settings()

    def test_s3_config_from_storage_s3(self, monkeypatch, tmp_path):
        """MINIO_PARAM and FS are initialized from storage.s3, not minio."""
        from hydrodatasource.configs import config

        data_root = tmp_path / "data"

        def mock_load():
            return {
                "storage": {
                    "local": {"root": str(data_root)},
                    "s3": {
                        "endpoint_url": "http://s3.example.com:9000",
                        "key": "testkey",
                        "secret": "testsecret",
                    },
                }
            }

        original_load = config._load_settings_from_file
        monkeypatch.setattr(config, "_load_settings_from_file", mock_load)
        config._init_settings()

        assert config.MINIO_PARAM["endpoint_url"] == "http://s3.example.com:9000"
        assert config.MINIO_PARAM["key"] == "testkey"
        assert config.MINIO_PARAM["secret"] == "testsecret"

        # Restore
        monkeypatch.setattr(config, "_load_settings_from_file", original_load)
        config._init_settings()

    def test_s3_config_empty_when_no_s3_section(self, monkeypatch, tmp_path):
        """MINIO_PARAM is empty dict, FS is None when no storage.s3."""
        from hydrodatasource.configs import config

        data_root = tmp_path / "data"

        def mock_load():
            return {"storage": {"local": {"root": str(data_root)}}}

        original_load = config._load_settings_from_file
        monkeypatch.setattr(config, "_load_settings_from_file", mock_load)
        config._init_settings()

        assert config.MINIO_PARAM == {}
        assert config.FS is None

        # Restore
        monkeypatch.setattr(config, "_load_settings_from_file", original_load)
        config._init_settings()

    def test_cache_dir_consistent_with_get_cache_dir(self, monkeypatch):
        """CACHE_DIR delegates to hydrodataset get_cache_dir."""
        from hydrodatasource.configs import config
        from pathlib import Path

        unified_cache = str(Path.home() / ".cache" / "unified_test")

        original_load = config._load_settings_from_file

        def mock_load():
            return {"storage": {"local": {"root": "/tmp/data"}}}

        monkeypatch.setattr(config, "_load_settings_from_file", mock_load)
        monkeypatch.setattr(config, "_hd_get_cache_dir", lambda: Path(unified_cache))
        config._init_settings()

        assert config.CACHE_DIR == unified_cache, (
            f"CACHE_DIR={config.CACHE_DIR} != expected {unified_cache}"
        )

        # Restore
        monkeypatch.setattr(config, "_load_settings_from_file", original_load)
        config._init_settings()

    def test_empty_config_creates_storage_format(self, monkeypatch):
        """When no config file exists, SETTING uses storage.* format, not local_data_path."""
        from hydrodatasource.configs import config

        def mock_load():
            return {}

        original_load = config._load_settings_from_file
        monkeypatch.setattr(config, "_load_settings_from_file", mock_load)
        config._init_settings()

        assert "local_data_path" not in config.SETTING, (
            "SETTING must not contain local_data_path after bridge removal"
        )
        assert "storage" in config.SETTING, (
            "SETTING must have storage key even when config file is empty"
        )
        assert config.SETTING["storage"]["local"]["root"], (
            "storage.local.root must have a fallback value"
        )

        # Restore
        monkeypatch.setattr(config, "_load_settings_from_file", original_load)
        config._init_settings()
