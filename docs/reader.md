# Reader

The `reader` module is the core component of `hydrodatasource` for accessing and reading various hydrological datasets. It provides a **unified, URI-only** interface for handling different data sources, with a special focus on custom, user-prepared datasets.

## Two Ways to Open a Dataset

### 1. The `open_dataset()` factory (registered datasets)

If a dataset is registered in the resolution registry (either in `HDS_DATASETS` or one of hydrodataset's datasets), you can resolve and open it in one line:

```python
from hydrodatasource.configs.data_resolver import open_dataset, resolve_data_path

# Registered dataset — reads ~/hydro_setting.yml for the storage root
ds = open_dataset("songliao_event")

# Or resolve just the path first
uri = resolve_data_path("songliao_event")
```

A custom `ResolverContext` can override the storage root and registry without touching `~/hydro_setting.yml`:

```python
from hydrodatasource.configs.data_resolver import open_dataset, ResolverContext

ctx = ResolverContext(storage={"local": {"root": "/your/data/root"}})
ds = open_dataset("songliao_event", ctx=ctx)
```

### 2. Direct URI-only construction (custom datasets)

For a dataset that is not registered, construct the reader class directly by passing the dataset directory as `uri`:

```python
from hydrodatasource.reader.data_source import SelfMadeHydroDataset

reader = SelfMadeHydroDataset(uri="/path/to/my_dataset", time_unit=["1D"])
```

> **Note.** Legacy `data_path=` / `dataset_name=` constructor arguments were removed and now raise `ValueError`.
> Pass the absolute path (or `s3://` URI) of the dataset directory as `uri`.

## Directory Structure

To use `SelfMadeHydroDataset`, your data should be organized in the following structure:

```
/path/to/my_dataset/
├── attributes/
│   ├── attributes.csv
├── shapes/
│   ├── basins.shp
├── timeseries/
│   ├── 1D/
│   │   ├── basin_1.csv
│   │   ├── basin_2.csv
│   │   ├── ...
│   ├── 1D_units_info.json
│   ├── 3h/
│   │   ├── basin_1.csv
│   │   ├── ...
│   ├── 3h_units_info.json
```

- **`attributes/attributes.csv`**: A CSV file containing static attributes for each basin (e.g., area, slope, land cover). It must contain a `basin_id` column.
- **`shapes/basins.shp`**: A shapefile containing the geographic boundaries of each basin.
- **`timeseries/`**: Time series data, with subdirectories for each time resolution (`1h`, `3h`, `1D`, `8D`, `1M`).
    - Each subdirectory contains CSV files, one for each basin, named with the `basin_id`.
    - Each subdirectory also contains a `*_units_info.json` file that specifies the units for the variables in the CSV files.

Extended readers may expect extra directories:

- **`intermediate/`** — interval-basin data with topology (`TgHydroDatasource`).
- **`stations/`** — gauging-station data and adjacency matrices (`StationHydroDataset`).
- **`forecasts/`** — forecast time series (`SelfMadeForecastDataset`).

## Example Usage

```python
from hydrodatasource.reader.data_source import SelfMadeHydroDataset

# Path to your dataset directory
reader = SelfMadeHydroDataset(uri="/path/to/my_dataset", time_unit=["1D"])

# Get a list of all basin IDs
basin_ids = reader.read_object_ids()

# Define the time range and variables to read
t_range = ["2000-01-01", "2010-12-31"]
variables = ["precipitation", "streamflow"]

# Read the time series data
timeseries_data = reader.read_ts_xrdataset(
    gage_id_lst=basin_ids,
    t_range=t_range,
    var_lst=variables,
    time_units=["1D"],
)

# The result is a dictionary with time units as keys and xarray.Dataset as values
daily_data = timeseries_data["1D"]
print(daily_data)
```

## Reader Aliases

All `hydrodatasource` readers are registered in `READER_ALIASES`:

| Alias | Class | Directory convention |
|-------|-------|---------------------|
| `selfmade` | `SelfMadeHydroDataset` | standard dataset |
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

hydrodataset's public datasets (e.g. `camels_us`) are also resolvable through the same
`open_dataset()` / `resolve_data_path()` interface.

## Other Readers

- **`SelfMadeForecastDataset`**: Extends `SelfMadeHydroDataset` to support forecast data, expected in a `forecasts` directory.
- **`StationHydroDataset`**: Extends `SelfMadeHydroDataset` to include data from gauging stations, expected in a `stations` directory.
- **`TgHydroDatasource`**: Extends `SelfMadeHydroDataset` with LSTM prediction and graph-network structure support, using an `intermediate/` directory for interval-basin topology.
