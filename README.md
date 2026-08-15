# hydrodatasource

[![image](https://img.shields.io/pypi/v/hydrodatasource.svg)](https://pypi.python.org/pypi/hydrodatasource) [![image](https://img.shields.io/conda/vn/conda-forge/hydrodatasource.svg)](https://anaconda.org/conda-forge/hydrodatasource) 

-   Free software: BSD license
-   Documentation: https://iHeadWater.github.io/hydrodatasource

## Overview

While libraries like [hydrodataset](https://github.com/OuyangWenyu/hydrodataset) exist for accessing standardized, public hydrological datasets (e.g., CAMELS), a common challenge is working with data that isn't in a ready-to-use format. This includes non-public industry data, data from local authorities, or custom datasets compiled for specific research projects.

**`hydrodatasource`** is designed to solve this problem. It provides a flexible framework to read, process, and clean these custom datasets, preparing them for hydrological modeling and analysis.

`hydrodatasource` uses a unified **URI-only** design: every reader is constructed by pointing it at a directory or an `s3://` URI, and a one-line `open_dataset()` factory resolves a dataset id to a ready-to-use reader. The design is compatible with [hydrodataset](https://github.com/OuyangWenyu/hydrodataset).

## Quick Start

If you have a **registered dataset** and a `~/hydro_setting.yml` (see [Configuration](#configuration)), resolve and open it in one line:

```python
from hydrodatasource.configs.data_resolver import open_dataset

# "songliao_event" is a hydrodatasource-registered dataset (flood-event reader)
ds = open_dataset("songliao_event")
```

To **open a custom dataset** by path, construct the reader directly:

```python
from hydrodatasource.reader.data_source import SelfMadeHydroDataset

reader = SelfMadeHydroDataset(uri="/path/to/my_dataset", time_unit=["1D"])
```

## Reading Custom Datasets

This is the primary use case for `hydrodatasource`. If you have your own basin-level time series and attribute data, you can use `SelfMadeHydroDataset` to load it seamlessly.

### 1. Prepare Your Data Directory

First, organize your data into the following folder structure:

```
/path/to/my_dataset/
├── attributes/
│   └── attributes.csv
├── shapes/
│   └── basins.shp
└── timeseries/
    ├── 1D/                     # Sub-folder for each time resolution (e.g., daily)
    │   ├── basin_01.csv
    │   ├── basin_02.csv
    │   └── ...
    └── 1D_units_info.json      # JSON file with unit information
```

-   **`attributes/attributes.csv`**: A CSV file containing static basin attributes (e.g., area, mean elevation). Must include a `basin_id` column that matches the filenames in the `timeseries` folder.
-   **`shapes/basins.shp`**: A shapefile with the polygon geometry for each basin.
-   **`timeseries/1D/`**: A folder for each time resolution (`1h`, `3h`, `1D`, `8D`, `1M`). Inside, each CSV file should contain the time series data for a single basin and be named after its `basin_id`.
-   **`timeseries/1D_units_info.json`**: A JSON file defining the units for each variable in your time series CSVs (e.g., `{"precipitation": "mm/d", "streamflow": "m3/s", "temperature": "degC"}`). Every variable you read must be listed here.

Extended datasets may add optional directories:

-   **`intermediate/`** — interval-basin (区间流域) data with topological relationships (`attributes/`, `timeseries/`, `shapes/`), used by the TG basin reader.
-   **`stations/`** — gauging-station data and adjacency matrices, used by the station reader.
-   **`forecasts/`** — forecast time series, used by the forecast reader.

### 2. Read the Data in Python

Once your data is organized, point a URI-only reader at it:

```python
from hydrodatasource.reader.data_source import SelfMadeHydroDataset

# Pass the absolute path (or s3:// URI) of your dataset directory as `uri`
reader = SelfMadeHydroDataset(uri="/path/to/my_dataset", time_unit=["1D"])

# Get a list of all available basin IDs
basin_ids = reader.read_object_ids()

# Define the time range and variables you want to load
t_range = ["2000-01-01", "2010-12-31"]
variables_to_read = ["precipitation", "streamflow", "temperature"]

# Read the time series data (a dict of xarray.Datasets keyed by time unit)
timeseries_data = reader.read_ts_xrdataset(
    gage_id_lst=basin_ids,
    t_range=t_range,
    var_lst=variables_to_read,
    time_units=["1D"],
)

daily_data = timeseries_data["1D"]

print("Successfully loaded data:")
print(daily_data)

# Static attributes are equally easy to read
attributes_data = reader.read_attr_xrdataset(gage_id_lst=basin_ids, var_lst=["area", "mean_elevation"])
print("\nAttributes:")
print(attributes_data)
```

> **Note on the old API.** Earlier versions accepted `data_path=` / `dataset_name=` constructor arguments.
> These were removed in favor of the unified `uri=` interface and now raise `ValueError`.

## Reader Aliases

All `hydrodatasource` readers are registered as aliases in `READER_ALIASES`, which downstream
projects (e.g. hydromodel) consume alongside hydrodataset's aliases:

| Alias | Class | Directory convention |
|-------|-------|---------------------|
| `selfmade` | `SelfMadeHydroDataset` | standard dataset (`attributes/`, `timeseries/`, `shapes/`) |
| `longterm` | `LongTermDataset` | self-made dataset with long-term support |
| `forecast` | `SelfMadeForecastDataset` | standard + `forecasts/` |
| `station` | `StationHydroDataset` | standard + `stations/` |
| `tghydro` | `TgHydroDatasource` | standard + `intermediate/` + LSTM predictions |
| `floodevent` | `FloodEventDatasource` | flood-event data with per-basin event markers |
| `gages` | `Gages` | GAGES-II public dataset |
| `grdc` | `Grdc` | GRDC public dataset |
| `rainfall` | `RainfallReader` | cleaned station rainfall |
| `crd` | `Crd` | China reservoir database |
| `rsvrinflow` | `RsvrInflowReader` | reservoir inflow data |

`hydrodataset`'s public datasets (e.g. `camels_us`) are also resolvable through the same
`open_dataset()` / `resolve_data_path()` interface.

## Configuration

`hydrodatasource` reads a shared `~/hydro_setting.yml` (same file used by `hydrodataset` and
`hydromodel`) using the unified `storage.*` format:

```yaml
storage:
  default_source: local      # 'local' or 'cloud' — fallback when source is not given
  local:
    root: 'D:\data\hydrodatasource'   # main data root
  cache: data\cache
  s3:                        # optional — cloud (MinIO/S3) access
    endpoint_url: 'http://minio:9000'
    key: 'access_key'
    secret: 'secret_key'
    bucket: hydro-data
    prefix: hydromodel
```

-   When `default_source: local`, `resolve_data_path()` / `open_dataset()` resolve against `storage.local.root`.
-   When `default_source: cloud`, they resolve against `storage.s3`.
-   If `~/hydro_setting.yml` is missing, a default root `~/hydrodatasource_data` is used (with a warning).

## Other Features

Beyond reading data, `hydrodatasource` also includes modules for:

-   **`processor`**: Perform advanced calculations like identifying rainfall-runoff events (`dmca_esr.py`) and calculating basin-wide mean rainfall from station data (`basin_mean_rainfall.py`).
-   **`cleaner`**: Clean raw time series data. This includes tools for smoothing noisy streamflow data, correcting anomalies in rainfall and water level records, and back-calculating reservoir inflow.

The usage of these modules is described in the [API Reference](https://iHeadWater.github.io/hydrodatasource/api). We will add more examples in the future.

## Installation

For standard use, install the package from PyPI:

```bash
pip install hydrodatasource
```

### Development Setup

For developers, it is recommended to use `uv` to manage the environment, as this project has local dependencies (e.g., `hydroutils>=0.2.0`, `hydrodataset>=0.3.0`).

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/iHeadWater/hydrodatasource.git
    cd hydrodatasource
    ```

2.  **Sync the environment with `uv`** (installs all extras, including dev tooling):
    ```bash
    uv sync --all-extras
    ```
