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

Critical globals (SETTING, CACHE_DIR, FS, LOCAL_ROOT, MINIO_PARAM) are
lazy-loaded via module __getattr__ to avoid import-time side effects.

FilePath: \\hydrodatasource\\hydrodatasource\\configs\\config.py
Copyright (c) 2023-2026 Wenyu Ouyang. All rights reserved.
"""

import os
import sys
import warnings
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


# ── Lazy-loading state ──────────────────────────────────────────────────────

_LAZY_KEYS = frozenset({"SETTING", "CACHE_DIR", "LOCAL_ROOT", "FS", "MINIO_PARAM"})
_lazy: Dict[str, Any] = {}
_initialized: bool = False


def _init_settings() -> None:
    """Load settings from ~/hydro_setting.yml — called lazily on first access."""
    global _initialized

    # Remove stale __dict__ entries that may have been left by monkeypatch
    # teardowns in tests (they shadow __getattr__ for lazy keys).
    for key in _LAZY_KEYS:
        globals().pop(key, None)

    try:
        setting = _load_settings_from_file()
    except (ValueError, FileNotFoundError) as e:
        warnings.warn(f"Could not load hydro_setting.yml: {e}", stacklevel=2)
        setting = {}

    default_root = os.path.join(Path.home(), "hydrodatasource_data")
    if not setting:
        warnings.warn(
            f"Using default data paths in home directory: {default_root}",
            stacklevel=2,
        )
        setting = {"storage": {"local": {"root": default_root}}}

    root = setting.get("storage", {}).get("local", {}).get("root", default_root)

    _lazy["SETTING"] = setting
    _lazy["LOCAL_ROOT"] = root
    _lazy["CACHE_DIR"] = str(_hd_get_cache_dir())
    # Ensure the cache dir exists: downstream code writes NetCDF caches and
    # lists this dir directly, and h5netcdf/os.listdir do not create it.
    os.makedirs(_lazy["CACHE_DIR"], exist_ok=True)

    # Initialize S3FS from storage.s3 credentials
    s3_cfg = setting.get("storage", {}).get("s3", {})
    if s3_cfg.get("endpoint_url") and s3_cfg.get("key"):
        _lazy["MINIO_PARAM"] = {
            "endpoint_url": s3_cfg["endpoint_url"],
            "key": s3_cfg["key"],
            "secret": s3_cfg.get("secret", ""),
        }
        try:
            _lazy["FS"] = s3fs.S3FileSystem(
                client_kwargs={"endpoint_url": _lazy["MINIO_PARAM"]["endpoint_url"]},
                key=_lazy["MINIO_PARAM"]["key"],
                secret=_lazy["MINIO_PARAM"]["secret"],
                use_ssl=False,
            )
        except Exception as e:
            warnings.warn(f"S3FS initialization failed: {e}", stacklevel=2)
            _lazy["FS"] = None
    else:
        _lazy["MINIO_PARAM"] = {}
        _lazy["FS"] = None

    _initialized = True


def __getattr__(name: str) -> Any:
    """Lazy-load critical globals on first access.

    Avoids import-time side effects (YAML reads, s3fs connections, print()
    calls) — all deferred until a caller actually accesses SETTING, CACHE_DIR,
    LOCAL_ROOT, FS, or MINIO_PARAM.
    """
    if name not in _LAZY_KEYS:
        raise AttributeError(
            f"module 'hydrodatasource.configs.config' has no attribute '{name}'"
        )
    if not _initialized:
        _init_settings()
    return _lazy[name]


# ── Public API (pure functions — no side effects) ─────────────────────────


def get_local_root() -> Optional[Path]:
    """Get the local storage root directory.

    Returns storage.local.root via hydrodataset settings.
    Falls back to LOCAL_ROOT if set.
    """
    root = _hd_get_local_root()
    if root is not None:
        return root
    local = getattr(sys.modules[__name__], "LOCAL_ROOT", "")
    if local:
        return Path(local)
    return None


def get_cache_dir() -> Path:
    """Get cache directory via hydrodataset's cache resolution logic."""
    return _hd_get_cache_dir()


def get_storage_config() -> Dict[str, Any]:
    """Get storage configuration block (new format)."""
    return _hd_get_storage_config()


def read_setting(setting_path: str) -> Dict[str, Any]:
    """Read and validate a hydro_setting.yml file.

    Only the storage.* format is accepted. Old local_data_path.* format is rejected.
    """
    if not os.path.exists(setting_path):
        raise FileNotFoundError(f"Configuration file not found: {setting_path}")

    with open(setting_path, "r", encoding="utf-8") as file:
        setting = yaml.safe_load(file)

    example_setting = (
        "# Required format:\n"
        "storage:\n"
        "  default_source: local\n"
        "  local:\n"
        "    root: 'D:\\\\data\\\\hydrodatasource'\n"
        "  cache: data\\\\cache\n"
        "  s3:\n"
        "    endpoint_url: 'http://minio:9000'\n"
        "    key: 'access_key'\n"
        "    secret: 'secret_key'\n"
        "    bucket: hydro-data\n"
        "    prefix: hydromodel\n"
    )

    if setting is None:
        raise ValueError(
            f"Configuration file is empty or has invalid format.\n\n"
            f"Example configuration:\n{example_setting}"
        )

    if "storage" not in setting:
        raise ValueError(
            f"Configuration must have 'storage' section.\n\n"
            f"Example configuration:\n{example_setting}"
        )

    return setting
