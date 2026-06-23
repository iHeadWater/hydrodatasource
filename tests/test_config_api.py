"""Tests for hydrodatasource.configs.config — new public API functions.

Tests the pure-function APIs (get_local_root, get_cache_dir, get_storage_config)
and read_setting() validation, plus backward-compatible globals.

These tests verify the changes made in feat/unified-data-interface.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml


# ── read_setting() tests ────────────────────────────────────────────────────


class TestReadSetting:
    """Verify read_setting() accepts both old and new config formats."""

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

    def test_old_format_local_data_path(self, tmp_path):
        """Old format with local_data_path.* is accepted (backward compat)."""
        from hydrodatasource.configs.config import read_setting

        setting_path = tmp_path / "hydro_setting.yml"
        setting_path.write_text(
            yaml.dump(
                {
                    "local_data_path": {
                        "root": str(tmp_path / "data"),
                        "datasets-origin": str(tmp_path / "data" / "datasets-origin"),
                        "datasets-interim": str(tmp_path / "data" / "datasets-interim"),
                    }
                }
            ),
            encoding="utf-8",
        )

        result = read_setting(str(setting_path))
        assert "local_data_path" in result

    def test_both_formats_accepted(self, tmp_path):
        """Config with both old and new sections is valid."""
        from hydrodatasource.configs.config import read_setting

        setting_path = tmp_path / "hydro_setting.yml"
        setting_path.write_text(
            yaml.dump(
                {
                    "storage": {
                        "local": {"root": str(tmp_path / "data")},
                    },
                    "local_data_path": {
                        "root": str(tmp_path / "data"),
                    },
                }
            ),
            encoding="utf-8",
        )

        result = read_setting(str(setting_path))
        assert "storage" in result
        assert "local_data_path" in result

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

    def test_missing_required_sections_raises(self, tmp_path):
        """Setting without 'storage' or 'local_data_path' raises ValueError."""
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

    def test_returns_path_when_local_data_path_set(self, monkeypatch):
        """get_local_root returns LOCAL_DATA_PATH when not None."""
        from hydrodatasource.configs import config

        monkeypatch.setattr(config, "LOCAL_DATA_PATH", "/test/root")
        # Patch _hd_get_local_root to return None (no new format)
        monkeypatch.setattr(config, "_hd_get_local_root", lambda: None)

        result = config.get_local_root()
        assert result == Path("/test/root")

    def test_returns_hd_root_when_available(self, monkeypatch):
        """get_local_root prefers hydrodataset's root when available."""
        from hydrodatasource.configs import config

        monkeypatch.setattr(
            config, "_hd_get_local_root", lambda: Path("/hd/root")
        )
        monkeypatch.setattr(config, "LOCAL_DATA_PATH", "/old/root")

        result = config.get_local_root()
        assert result == Path("/hd/root")

    def test_returns_none_when_nothing_configured(self, monkeypatch):
        """get_local_root returns None when nothing is configured."""
        from hydrodatasource.configs import config

        monkeypatch.setattr(config, "_hd_get_local_root", lambda: None)
        monkeypatch.setattr(config, "LOCAL_DATA_PATH", "")

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

    def test_local_data_path_is_str(self):
        """LOCAL_DATA_PATH is a str (may be empty)."""
        from hydrodatasource.configs.config import LOCAL_DATA_PATH

        assert isinstance(LOCAL_DATA_PATH, str)

    def test_cache_dir_is_str(self):
        """CACHE_DIR is a str (may be empty)."""
        from hydrodatasource.configs.config import CACHE_DIR

        assert isinstance(CACHE_DIR, str)

    def test_minio_param_is_dict(self):
        """MINIO_PARAM is a dict (may be empty)."""
        from hydrodatasource.configs.config import MINIO_PARAM

        assert isinstance(MINIO_PARAM, dict)


# ── Format bridging tests ───────────────────────────────────────────────────


class TestFormatBridging:
    """Verify old-format and new-format interop in _init_settings."""

    def test_new_format_derives_local_data_path(self, monkeypatch, tmp_path):
        """When only storage.* is set, local_data_path is derived."""
        from hydrodatasource.configs import config
        import importlib

        # Create a temp setting file with only new format
        setting_path = Path.home() / "hydro_setting.yml"
        # We can't modify the real home config, so test the logic directly
        # by calling _init_settings with mocked _load_settings_from_file
        data_root = tmp_path / "data"

        original_load = config._load_settings_from_file

        def mock_load():
            return {
                "storage": {
                    "default_source": "local",
                    "local": {"root": str(data_root)},
                }
            }

        monkeypatch.setattr(config, "_load_settings_from_file", mock_load)
        config._init_settings()

        assert config.LOCAL_DATA_PATH == str(data_root)

        # Restore
        monkeypatch.setattr(config, "_load_settings_from_file", original_load)
        config._init_settings()

    def test_old_format_still_works(self, monkeypatch, tmp_path):
        """Old local_data_path format still initializes correctly."""
        from hydrodatasource.configs import config

        data_root = str(tmp_path / "data")

        def mock_load():
            return {
                "local_data_path": {
                    "root": data_root,
                    "datasets-origin": os.path.join(data_root, "datasets-origin"),
                    "datasets-interim": os.path.join(data_root, "datasets-interim"),
                    "cache": os.path.join(data_root, ".cache"),
                }
            }

        original_load = config._load_settings_from_file
        monkeypatch.setattr(config, "_load_settings_from_file", mock_load)
        config._init_settings()

        assert config.LOCAL_DATA_PATH == data_root

        # Restore
        monkeypatch.setattr(config, "_load_settings_from_file", original_load)
        config._init_settings()
