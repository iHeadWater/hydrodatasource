"""Guard tests for three review Suggestion-level issues (RED on current code).

These are reproduction tests for bugs surfaced in a code review pass. Each
test exercises the *unfixed* behaviour and must FAIL on the current source,
acting as the regression guard for the implementer's fix.

S5 -- ``hydrodatasource/utils/utils.py::is_minio_folder`` missing ``FS is None``
    guard. Its sibling ``minio_file_list`` already tolerates an unconfigured S3
    backend (``if FS is None: return []``), but ``is_minio_folder`` calls
    ``FS.exists(...)`` unconditionally, so with S3 unconfigured it raises
    ``AttributeError: 'NoneType' object has no attribute 'exists'`` instead of
    reporting the path is not a MinIO folder. Expected fix: return ``False``
    when ``FS`` is ``None``.

S4 -- ``hydrodatasource/configs/config.py::_init_settings`` does not
    normalise ``SETTING`` when the settings dict is non-empty but lacks a
    ``storage`` key. The empty-settings branch injects
    ``{"storage": {"local": {"root": ...}}}``, but the ``elif "storage" not in
    setting`` branch only warns and leaves the dict without ``storage``, so
    downstream ``SETTING["storage"]`` raises ``KeyError``. Expected fix: the
    elif branch must inject the same ``storage.local.root`` fallback.

S6 -- ``hydrodatasource/reader/data_source.py::LongTermDataset`` exposes a dead
    ``download`` constructor parameter. ``__init__`` accepts ``download=False``
    but never reads it. Expected fix: remove the parameter (no in-repo caller
    passes it). Test asserts the constructor signature no longer contains it.

These tests deliberately avoid the ``internal_data`` marker and keep every
import inside the test function, so collection is side-effect free.
"""

# ── S5: is_minio_folder must tolerate FS = None (S3 unconfigured) ───────────


def test_is_minio_folder_returns_false_when_s3_not_configured(monkeypatch):
    """is_minio_folder returns False (not AttributeError) when FS is None.

    The current code calls ``FS.exists(minio_url)`` unconditionally; with S3
    unconfigured ``FS`` is ``None`` and the call dies with AttributeError. The
    expected behaviour, matching ``minio_file_list``'s ``if FS is None: return
    []``, is that an unconfigured S3 backend means no path is a MinIO folder.
    """
    import hydrodatasource.utils.utils as utils

    # Patch the module-level binding in utils' own namespace. utils.py does
    # ``from ..configs.config import FS`` at import time, so patching
    # config.FS would NOT reach the global ``FS`` that is_minio_folder reads
    # (the module-binding trap) -- we must patch utils.FS directly.
    monkeypatch.setattr(utils, "FS", None)

    assert utils.is_minio_folder("s3://bucket/some/dir") is False


# ── S4: _init_settings must normalise SETTING when 'storage' is missing ─────


def test_init_settings_injects_storage_defaults_when_storage_section_missing(
    monkeypatch,
):
    """A non-empty settings dict without a 'storage' key still yields SETTING['storage'].

    The empty-settings branch injects ``{"storage": {"local": {"root": ...}}}``,
    but the elif branch (non-empty, no 'storage' key) only warns and leaves the
    dict without 'storage' -- so ``SETTING["storage"]`` KeyErrors downstream. The
    expected fix normalises the dict just like the empty branch does.
    """
    from hydrodatasource.configs import config

    original_load = config._load_settings_from_file

    def mock_load():
        # Non-empty but missing the 'storage' section.
        return {"project": "demo", "some": "legacy-key"}

    monkeypatch.setattr(config, "_load_settings_from_file", mock_load)
    try:
        config._init_settings()

        storage = config.SETTING["storage"]  # KeyError on current code
        assert storage["local"]["root"], (
            "storage.local.root must have a fallback value when the settings "
            "file has no storage section"
        )
    finally:
        # Restore the real loader and re-initialize so other tests see a
        # genuine config even when this test fails (RED).
        monkeypatch.setattr(config, "_load_settings_from_file", original_load)
        config._init_settings()


# ── S6: LongTermDataset must not expose the dead 'download' parameter ───────


def test_longterm_dataset_init_has_no_download_parameter():
    """LongTermDataset.__init__ does not accept a dead 'download' argument.

    The constructor declares ``download=False`` but never reads it, so it is
    pure dead weight on the unified URI-only constructor contract. No in-repo
    caller passes ``download=`` (verified by searching the repository), so the
    parameter can be dropped outright.
    """
    import inspect

    from hydrodatasource.reader.data_source import LongTermDataset

    params = list(inspect.signature(LongTermDataset.__init__).parameters)
    assert "download" not in params
