"""
Author: Jianfeng Zhu, Wenyu Ouyang
Date: 2023-10-25 18:49:02
LastEditTime: 2025-06-18
LastEditors: Wenyu Ouyang
Description: Storage and cache configuration for hydrodatasource.

Reads ~/hydro_setting.yml (shared with hydromodel and hydrodataset).
Supports the unified storage.* format defined in hydromodel ADR 0001.

Legacy minio.* and postgres.* config blocks have been removed.
Cloud access is unified under storage.s3.
Database access is decoupled to a separate real-time service.

Critical globals (SETTING, CACHE_DIR, FS, LOCAL_DATA_PATH, MINIO_PARAM) are
initialized at import time for backward compatibility.
New code should use the pure-function APIs (get_local_root, get_cache_dir).

FilePath: \\hydrodatasource\\hydrodatasource\\configs\\config.py
Copyright (c) 2023-2026 Wenyu Ouyang. All rights reserved.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import s3fs
import yaml

from hydrodataset.configs.settings import (
    get_cache_dir as _hd_get_cache_dir,
    get_local_root as _hd_get_local_root,
    get_storage_config as _hd_get_storage_config,
)

# ── Internal helpers (defined first — called by init) ───────────────────────


def _load_settings_from_file() -> Dict[str, Any]:
    """Load settings from ~/hydro_setting.yml (supports old and new formats)."""
    setting_path = os.path.join(Path.home(), "hydro_setting.yml")
    if not os.path.exists(setting_path):
        return {}

    with open(setting_path, "r", encoding="utf-8") as file:
        setting = yaml.safe_load(file)

    if setting is None:
        return {}

    return setting


# ── Module-level state (initialized once at import for backward compat) ──

SETTING: Dict[str, Any] = {}
CACHE_DIR: str = ""
LOCAL_DATA_PATH: str = ""
FS: Optional[s3fs.S3FileSystem] = None
MINIO_PARAM: Dict[str, str] = {}


def _init_settings() -> None:
    """Load settings from ~/hydro_setting.yml once at module load time."""
    global SETTING, CACHE_DIR, LOCAL_DATA_PATH, FS, MINIO_PARAM

    try:
        setting = _load_settings_from_file()
    except (ValueError, FileNotFoundError) as e:
        print(f"Warning: Could not load hydro_setting.yml: {e}")
        setting = {}

    default_root = os.path.join(Path.home(), "hydrodatasource_data")
    if not setting:
        print(f"Using default data paths in home directory: {default_root}")
        setting = {
            "local_data_path": {
                "root": default_root,
                "datasets-origin": os.path.join(default_root, "datasets-origin"),
                "datasets-interim": os.path.join(default_root, "datasets-interim"),
                "cache": os.path.join(default_root, ".cache"),
            }
        }

    # Bridge: if new format (storage.*) exists but old (local_data_path) is
    # missing, create derived local_data_path for backward compat.
    if "local_data_path" not in setting and "storage" in setting:
        storage_local = setting["storage"].get("local", {})
        storage_root = storage_local.get("root", default_root)
        setting["local_data_path"] = {
            "root": storage_root,
            "datasets-origin": os.path.join(storage_root, "datasets-origin"),
            "datasets-interim": os.path.join(storage_root, "datasets-interim"),
            "cache": storage_local.get(
                "cache", os.path.join(storage_root, ".cache")
            ),
        }

    root = setting.get("local_data_path", {}).get("root", default_root)
    cache = setting.get("local_data_path", {}).get(
        "cache", os.path.join(Path.home(), "hydrodatasource_data", ".cache")
    )

    SETTING = setting
    LOCAL_DATA_PATH = root
    CACHE_DIR = cache

    # Initialize S3FS only if minio credentials are present in old-format config
    minio_cfg = setting.get("minio", {})
    if minio_cfg.get("client_endpoint") and minio_cfg.get("access_key"):
        MINIO_PARAM = {
            "endpoint_url": minio_cfg["client_endpoint"],
            "key": minio_cfg["access_key"],
            "secret": minio_cfg.get("secret", ""),
        }
        try:
            FS = s3fs.S3FileSystem(
                client_kwargs={"endpoint_url": MINIO_PARAM["endpoint_url"]},
                key=MINIO_PARAM["key"],
                secret=MINIO_PARAM["secret"],
                use_ssl=False,
            )
        except Exception as e:
            print(f"Warning: S3FS initialization failed: {e}")
            FS = None
    else:
        MINIO_PARAM = {}
        FS = None


# Initialize at module load (backward compat)
_init_settings()


# ── Public API (pure functions — no side effects) ─────────────────────────


def get_local_root() -> Optional[Path]:
    """Get the local storage root directory.

    Uses the new storage.local.root format (via hydrodataset settings).
    Falls back to the old local_data_path.root format for backward compat.
    """
    root = _hd_get_local_root()
    if root is not None:
        return root
    if LOCAL_DATA_PATH:
        return Path(LOCAL_DATA_PATH)
    return None


def get_cache_dir() -> Path:
    """Get cache directory via hydrodataset's cache resolution logic."""
    return _hd_get_cache_dir()


def get_storage_config() -> Dict[str, Any]:
    """Get storage configuration block (new format)."""
    return _hd_get_storage_config()


def read_setting(setting_path: str) -> Dict[str, Any]:
    """Read and validate a hydro_setting.yml file.

    Accepts both old (local_data_path.*) and new (storage.*) formats.
    No longer requires minio or postgres sections.
    """
    if not os.path.exists(setting_path):
        raise FileNotFoundError(f"Configuration file not found: {setting_path}")

    with open(setting_path, "r", encoding="utf-8") as file:
        setting = yaml.safe_load(file)

    example_setting = (
        "# New format (recommended):\n"
        "storage:\n"
        "  default_source: local\n"
        "  local:\n"
        "    root: 'D:\\\\data\\\\hydrodatasource'\n"
        "  s3:\n"
        "    bucket: hydro-data\n"
        "    prefix: hydromodel\n"
        "    region: us-east-1\n"
        "    profile: default\n\n"
        "# Old format (deprecated, still supported):\n"
        "local_data_path:\n"
        "  root: 'D:\\\\data\\\\waterism'\n"
        "  datasets-origin: 'D:\\\\data\\\\waterism\\\\datasets-origin'\n"
        "  datasets-interim: 'D:\\\\data\\\\waterism\\\\datasets-interim'\n"
        "  cache: 'D:\\\\data\\\\waterism\\\\.cache'\n"
    )

    if setting is None:
        raise ValueError(
            f"Configuration file is empty or has invalid format.\n\n"
            f"Example configuration:\n{example_setting}"
        )

    has_old = "local_data_path" in setting
    has_new = "storage" in setting
    if not has_old and not has_new:
        raise ValueError(
            f"Configuration must have 'storage' (new format) or "
            f"'local_data_path' (old format) section.\n\n"
            f"Example configuration:\n{example_setting}"
        )

    return setting
