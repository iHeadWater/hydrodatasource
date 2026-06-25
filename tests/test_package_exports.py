"""Tests for hydrodatasource package-level exports (__init__.py).

Verifies that READER_ALIASES, resolve_data_path, and DatasetResolutionError
are importable from the top-level package.
"""

import pytest


class TestPackageExports:
    """Verify new exports in hydrodatasource.__init__."""

    def test_reader_aliases_importable_from_package(self):
        """READER_ALIASES is importable from hydrodatasource."""
        from hydrodatasource import READER_ALIASES

        assert isinstance(READER_ALIASES, dict)
        assert len(READER_ALIASES) >= 11

    def test_resolve_data_path_importable_from_package(self):
        """resolve_data_path is importable from hydrodatasource."""
        from hydrodatasource import resolve_data_path

        assert callable(resolve_data_path)

    def test_dataset_resolution_error_importable_from_package(self):
        """DatasetResolutionError is importable from hydrodatasource."""
        from hydrodatasource import DatasetResolutionError

        assert issubclass(DatasetResolutionError, ValueError)

    def test_merged_aliases_include_hydrodataset_entries(self):
        """Merged READER_ALIASES includes entries from hydrodataset."""
        from hydrodatasource import READER_ALIASES

        # hydrodataset entries should be present (these are from the 37 public datasets)
        assert "camels_us" in READER_ALIASES, "hydrodataset aliases not merged"
        hd_entry = READER_ALIASES["camels_us"]
        assert hd_entry["category"] == "hydrodataset"

    def test_merged_aliases_include_hydrodatasource_entries(self):
        """Merged READER_ALIASES includes entries from hydrodatasource."""
        from hydrodatasource import READER_ALIASES

        assert "floodevent" in READER_ALIASES
        hds_entry = READER_ALIASES["floodevent"]
        assert hds_entry["category"] == "hydrodatasource"

    def test_hydrodatasource_aliases_total_count(self):
        """hydrodatasource has exactly 11 reader aliases."""
        from hydrodatasource import READER_ALIASES

        hds_aliases = {
            k: v for k, v in READER_ALIASES.items() if v.get("category") == "hydrodatasource"
        }
        assert len(hds_aliases) == 11, (
            f"Expected 11 hydrodatasource aliases, got {len(hds_aliases)}: "
            f"{sorted(hds_aliases.keys())}"
        )

    def test_merged_total_count(self):
        """Total merged aliases: 11 hydrodatasource + 26 hydrodataset = 37."""
        from hydrodatasource import READER_ALIASES

        assert len(READER_ALIASES) == 37, (
            f"Expected 37 total aliases (11 HDS + 26 HD), got {len(READER_ALIASES)}"
        )

    def test_resolver_context_importable_from_package(self):
        """ResolverContext is importable from hydrodatasource."""
        from hydrodatasource import ResolverContext
        from dataclasses import is_dataclass

        assert is_dataclass(ResolverContext)
