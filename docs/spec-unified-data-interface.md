# Spec: Unified Data Interface Alignment (hydrodatasource)

## Status

Approved — ready for implementation.

## Objective

Align `hydrodatasource` with `hydromodel`'s [ADR 0001: Unified Data Path Resolution](../hydromodel/docs/adr/0001-unified-data-path-resolution.md)
and `hydrodataset`'s configuration system, so all three repositories share one deterministic
contract for resolving dataset identity → storage location → runtime URI.

**Key principles:**
- hydrodatasource should reuse hydrodataset's config parsing code — NOT reinvent wheels
- hydrodatasource contributes its own **reader aliases** and **dataset registry**, which hydromodel consumes alongside hydrodataset's
- Database and legacy MinIO configuration must be **removed** — storage is unified under `storage.local` / `storage.s3` (cloud)
- All hydrodatasource reader classes must be registered as reader aliases

## Resolved Questions

| Question | Answer |
|----------|--------|
| Explicit hydrodataset dependency? | **Yes** — add to pyproject.toml since we import its settings.py and data_resolver.py directly |
| Old minio config? | **Remove** — superseded by `storage.s3` (cloud) |
| Old postgres config? | **Remove** — database access belongs in a real-time service repo, not coupled here |
| hydromodel dataset_dict.py? | **Deprecate/remove** — superseded by merged READER_ALIASES from both packages |
| Which readers to register? | **All of them** — every HydroData subclass gets a reader alias |

## Current State

### What exists today

| Repo | data_resolver | datasets.yml | READER_ALIASES | settings.py |
|------|--------------|-------------|----------------|-------------|
| hydromodel | `hydromodel/configs/data_resolver.py` | `configs/datasets.yml` (2 datasets) | 4 entries (hardcoded) | built into data_resolver |
| hydrodataset | `hydrodataset/configs/data_resolver.py` | `configs/datasets.yml` (37 datasets) | 37 entries | `hydrodataset/configs/settings.py` |
| hydrodatasource | **NONE** | **NONE** | **NONE** | Old-style `configs/config.py` with import-time side effects |

### The gap

```
OLD (current hydrodatasource):
  ~/hydro_setting.yml (local_data_path.root, minio.*, postgres.*)
    → configs/config.py (import-time side effects: SETTING, CACHE_DIR, FS, PS)
      → reader classes (data_path + dataset_name)

NEW (target):
  ~/hydro_setting.yml (storage.local.root, storage.s3.* — NO minio, NO postgres)
    → hydrodataset.configs.settings (pure functions)
      + hydrodatasource.configs.data_resolver (READER_ALIASES + resolve_data_path)
        + configs/datasets.yml (dataset id → reader + relative path)
          → reader classes (accept resolved URI)
```

### Problems to fix

1. Import-time side effects in `configs/config.py`
2. Old config format (`local_data_path.root` vs new `storage.local.root`)
3. No dataset registry (no `datasets.yml`)
4. No reader aliases for hydrodatasource's readers
5. Path-based constructors instead of URI-accepting ones
6. **minio config block — REMOVE** (cloud access via `storage.s3` only)
7. **postgres config block — REMOVE** (database decoupled from this repo)

## Tech Stack

- Python 3.10+
- PyYAML for config parsing
- pathlib for path operations
- **hydrodataset** as explicit dependency in pyproject.toml

## Commands

```bash
uv sync --dev
uv run pytest tests/ -v
uv run black hydrodatasource/
uv run flake8 hydrodatasource/
uv build
```

## Design Decisions

### Decision 1: Reuse hydrodataset's settings.py directly

hydrodatasource will import `get_local_root`, `get_cache_dir`, `load_settings` from
`hydrodataset.configs.settings`.

### Decision 2: hydrodatasource gets its own data_resolver.py

`hydrodatasource/configs/data_resolver.py` with:
- `READER_ALIASES` for ALL hydrodatasource reader classes
- `resolve_data_path()` delegating to hydrodataset's logic
- `DatasetResolutionError` re-exported from hydrodataset

### Decision 3: Two-stage migration for config.py

**Stage A (this spec):** Add new functions, keep `CACHE_DIR`/`SETTING` globals via lazy compat layer. Remove minio/postgres globals.
**Stage B (future):** Deprecate and remove all import-time side effects.

### Decision 4: Remove minio and postgres config blocks

The `~/hydro_setting.yml` for hydrodatasource must NOT contain `minio.*` or `postgres.*`.
Cloud storage uses `storage.s3.*`. Database access is decoupled to a separate real-time service.

### Decision 5: Register ALL reader classes

All 11 HydroData subclasses get a reader alias.

### Decision 6: READER_ALIASES format follows hydrodataset

```python
READER_ALIASES: Dict[str, Dict[str, str]] = {
    "alias_name": {
        "module": "hydrodatasource.reader.floodevent",
        "class": "FloodEventDatasource",
        "category": "hydrodatasource",
    },
}
```

## Implementation Phases

### Phase 1: Create data_resolver.py for hydrodatasource

**File:** `hydrodatasource/configs/data_resolver.py`

Register ALL reader classes:

| Alias | Module | Class |
|-------|--------|-------|
| `floodevent` | hydrodatasource.reader.floodevent | FloodEventDatasource |
| `selfmade` | hydrodatasource.reader.data_source | SelfMadeHydroDataset |
| `longterm` | hydrodatasource.reader.data_source | LongTermDataset |
| `forecast` | hydrodatasource.reader.data_source | SelfMadeForecastDataset |
| `station` | hydrodatasource.reader.data_source | StationHydroDataset |
| `tghydro` | hydrodatasource.reader.data_source | TgHydroDatasource |
| `gages` | hydrodatasource.reader.gages | Gages |
| `grdc` | hydrodatasource.reader.grdc | Grdc |
| `rainfall` | hydrodatasource.reader.rainfall_reader | RainfallReader |
| `crd` | hydrodatasource.reader.reservoir_datasets | Crd |
| `rsvrinflow` | hydrodatasource.reader.rsvr_inflow_reader | RsvrInflowReader |

Also implements `resolve_data_path(dataset_id, *, source, project_root)` and re-exports `DatasetResolutionError`.

### Phase 2: Create top-level configs/datasets.yml

**File:** `configs/datasets.yml` (repo root)

### Phase 3: Refactor configs/config.py — additive + remove minio/postgres

- Add lazy loading via `__getattr__` at module level
- Add `get_local_root()` → delegates to `hydrodataset.configs.settings`
- Add `get_cache_dir()` → same
- Keep `CACHE_DIR`, `SETTING`, `LOCAL_DATA_PATH` as deprecated compat
- **Remove:** `MINIO_PARAM`, `RO`, `S3`, `MC`, `FS` globals
- **Remove:** `POSTGRES_PARAM`, `PS`, `get_postgres_connection()`
- **Remove:** `read_setting()` validation of `minio` and `postgres` sections

### Phase 4: Add URI-accepting constructors to reader classes

Update `HydroData.__init__` and subclasses to accept `uri` parameter.

### Phase 5: Update hydromodel data_resolver.py

Import and merge READER_ALIASES from both hydrodataset and hydrodatasource.
Deprecate `dataset_dict.py`.

### Phase 6: Add hydrodataset dependency to pyproject.toml

## Success Criteria

- [ ] `hydrodatasource/configs/data_resolver.py` with READER_ALIASES for ALL 11 reader classes
- [ ] `configs/datasets.yml` at repo root
- [ ] `resolve_data_path("songliao_event")` resolves correctly
- [ ] `from hydrodatasource.configs.config import CACHE_DIR` still works
- [ ] No `minio` or `postgres` config validation in config.py
- [ ] `SelfMadeHydroDataset(uri="/data/...")` works
- [ ] hydromodel can import and merge hydrodatasource's READER_ALIASES
- [ ] No import-time side effects in new code paths
- [ ] Same `~/hydro_setting.yml` (`storage.local.root`) works for all three repos
- [ ] hydrodataset is an explicit dependency in pyproject.toml

## Boundaries

- **Always do:** Use hydrodataset's settings.py; define reader aliases in Python code; validate paths per ADR 0001; use `storage.local.root` format
- **Ask first:** Removing old `config.py` globals entirely; changing reader class constructor signatures incompatibly
- **Never do:** Define Python module import paths in YAML; create datasets with absolute paths; allow `..` in dataset paths; commit secrets; add `minio.*` or `postgres.*` config validation back
