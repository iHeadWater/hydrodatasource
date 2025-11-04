# hydrodatasource

[![image](https://img.shields.io/pypi/v/hydrodatasource.svg)](https://pypi.python.org/pypi/hydrodatasource) [![image](https://img.shields.io/conda/vn/conda-forge/hydrodatasource.svg)](https://anaconda.org/conda-forge/hydrodatasource) 

-   Free software: BSD license
-   Documentation: https://iHeadWater.github.io/hydrodatasource

## Overview

While libraries like [hydrodataset](https://github.com/OuyangWenyu/hydrodataset) exist for accessing standardized, public hydrological datasets (e.g., CAMELS), a common challenge is working with data that isn't in a ready-to-use format. This includes non-public industry data, data from local authorities, or custom datasets compiled for specific research projects.

**`hydrodatasource`** is designed to solve this problem. It provides a flexible framework to read, process, and clean these custom datasets, preparing them for hydrological modeling and analysis.

The core of this framework is the `SelfMadeHydroDataset` class, which allows you to easily access your own data by organizing it into a simple, predefined directory structure.

## Reading Custom Datasets with `SelfMadeHydroDataset`

This is the primary use case for `hydrodatasource`. If you have your own basin-level time series and attribute data, you can use this class to load it seamlessly.

### 1. Prepare Your Data Directory

First, organize your data into the following folder structure:

```
/path/to/your_data_root/
    └── my_custom_dataset/              # Your dataset's name
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
-   **`timeseries/1D/`**: A folder for each time resolution (e.g., `1D` for daily, `3h` for 3-hourly). Inside, each CSV file should contain the time series data for a single basin and be named after its `basin_id`.
-   **`timeseries/1D_units_info.json`**: A JSON file defining the units for each variable in your time series CSVs (e.g., `{"precipitation": "mm/d", "streamflow": "m3/s"}`).

### 2. Read the Data in Python

Once your data is organized, you can use `SelfMadeHydroDataset` to read it with just a few lines of code.

```python
from hydrodatasource.reader.data_source import SelfMadeHydroDataset

# 1. Define the path to your data's parent directory and the dataset name
data_path = "/path/to/your_data_root/"
dataset_name = "my_custom_dataset"

# 2. Initialize the reader
# Specify the time units you want to work with
reader = SelfMadeHydroDataset(data_path=data_path, dataset_name=dataset_name, time_unit=["1D"])

# 3. Get a list of all available basin IDs
basin_ids = reader.read_object_ids()

# 4. Define the time range and variables you want to load
t_range = ["2000-01-01", "2010-12-31"]
variables_to_read = ["precipitation", "streamflow", "temperature"]

# 5. Read the time series data
# The result is a dictionary of xarray.Datasets, keyed by time unit
timeseries_data = reader.read_ts_xrdataset(
    gage_id_lst=basin_ids,
    t_range=t_range,
    var_lst=variables_to_read,
    time_units=["1D"]
)

daily_data = timeseries_data["1D"]

print("Successfully loaded data:")
print(daily_data)

# You can also read the static attributes
attributes_data = reader.read_attr_xrdataset(gage_id_lst=basin_ids, var_lst=["area", "mean_elevation"])
print("\nAttributes:")
print(attributes_data)
```

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

For developers, it is recommended to use `uv` to manage the environment, as this project has local dependencies (e.g., `hydroutils`, `hydrodataset`).

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/iHeadWater/hydrodatasource.git
    cd hydrodatasource
    ```

2.  **Sync the environment with `uv`:**
    This command will install all dependencies, including the local editable packages.
    ```bash
    uv sync --all-extras
    ```