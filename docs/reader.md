# Reader

The `reader` module is the core component of `hydrodatasource` for accessing and reading various hydrological datasets. It provides a unified interface for handling different data sources, with a special focus on custom, user-prepared datasets.

## SelfMadeHydroDataset

The `SelfMadeHydroDataset` class is the most important feature of the `reader` module. It allows you to read your own hydrological data as long as it follows a specific directory structure. This is designed for flexibility, enabling you to work with non-public or specially prepared datasets.

### Directory Structure

To use `SelfMadeHydroDataset`, your data should be organized in the following structure:

```
/path/to/your_dataset_name/
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
- **`timeseries/`**: This directory holds the time series data, with subdirectories for each time resolution (e.g., `1D` for daily, `3h` for 3-hourly).
    - Each subdirectory contains CSV files, one for each basin, named with the `basin_id`.
    - Each subdirectory also contains a `*_units_info.json` file that specifies the units for the variables in the CSV files.

### Example Usage

Here is how you can use `SelfMadeHydroDataset` to read your data:

```python
from hydrodatasource.reader.data_source import SelfMadeHydroDataset

# Path to the parent directory of your dataset
data_path = "/path/to/your_data/"
# The name of your dataset directory
dataset_name = "my_custom_dataset"

# Initialize the reader
reader = SelfMadeHydroDataset(data_path=data_path, dataset_name=dataset_name, time_unit=["1D"])

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
    time_units=["1D"]
)

# The result is a dictionary with time units as keys and xarray.Dataset as values
daily_data = timeseries_data["1D"]
print(daily_data)
```

## Other Readers

- **`SelfMadeForecastDataset`**: Extends `SelfMadeHydroDataset` to support forecast data, which is expected to be in a `forecasts` directory.
- **`StationHydroDataset`**: Extends `SelfMadeHydroDataset` to include data from gauging stations, which is expected to be in a `stations` directory.
