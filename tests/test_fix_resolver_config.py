"""RED regression tests for two resolver/config defects.

These tests document the *desired* behavior and are expected to FAIL against
the current implementation (TDD RED phase / GAN discriminator).

Defect A — `source` default bypasses `storage.default_source`
    hydrodatasource.configs.data_resolver.resolve_data_path / open_dataset
    hardcode ``source: Literal["local", "cloud"] = "local"``, so they always
    forward an explicit ``"local"`` to the underlying hydrodataset resolver.
    The underlying resolver's fallback to ``storage.default_source`` (when
    ``source is None``) is therefore never reached.

Defect B — non-empty settings without a ``storage`` key are silently ignored
    hydrodatasource.configs.config._init_settings only warns when the loaded
    settings dict is *empty*.  A non-empty dict that lacks the ``storage``
    key silently falls back to the default local root with no UserWarning.
"""

import pytest

from hydrodatasource.configs import config as config_module
from hydrodatasource.configs.data_resolver import (
    ResolverContext,
    resolve_data_path,
)


def _make_local_dataset_dir(tmp_path):
    """Create ``{root}/projects/songliao/event`` under *tmp_path*.

    ``_resolve_local`` requires both ``storage.local.root`` and the resolved
    registry-relative directory to exist on disk.
    """
    dataset_dir = tmp_path / "projects" / "songliao" / "event"
    dataset_dir.mkdir(parents=True)
    return dataset_dir


class TestResolveDataPathDefaultSource:
    """Defect A: default source must follow ``storage.default_source``."""

    def test_default_cloud_returns_s3_uri(self, tmp_path):
        """Omitted source + ``default_source: cloud`` must yield an s3:// URI."""
        ctx = ResolverContext(
            storage={"default_source": "cloud", "s3": {"bucket": "b", "prefix": "p"}},
            project_root=tmp_path,
        )

        uri = resolve_data_path("songliao_event", ctx=ctx)

        assert uri.startswith("s3://")

    def test_default_local_returns_local_path(self, tmp_path):
        """Omitted source + ``default_source: local`` must yield a local path."""
        dataset_dir = _make_local_dataset_dir(tmp_path)
        ctx = ResolverContext(
            storage={"default_source": "local", "local": {"root": str(tmp_path)}},
            project_root=tmp_path,
        )

        uri = resolve_data_path("songliao_event", ctx=ctx)

        assert not uri.startswith("s3://")
        assert uri == str(dataset_dir)


class TestInitSettingsMissingStorageWarning:
    """Defect B: non-empty settings lacking a ``storage`` key must warn."""

    def test_non_empty_missing_storage_warns(self, monkeypatch):
        """A UserWarning mentioning 'storage' is expected when settings lack it."""
        monkeypatch.setattr(config_module, "_initialized", False)
        monkeypatch.setattr(
            config_module, "_load_settings_from_file", lambda: {"foo": "bar"}
        )

        with pytest.warns(UserWarning, match="storage"):
            config_module._init_settings()
