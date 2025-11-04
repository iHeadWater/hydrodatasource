# Processor

The `processor` module contains functions for advanced processing of hydrological data. This includes spatial analysis, like calculating basin-average rainfall, and time series analysis, like identifying distinct rainfall-runoff events.

## Basin Mean Rainfall

When working with multiple rainfall gauges in a basin, you often need to calculate a single, representative rainfall value for the entire basin. The `basin_mean_rainfall.py` module provides tools for this.

- **`calculate_thiesen_polygons`**: This function generates Thiessen polygons from station locations. Each polygon represents the area that is closest to a particular station, and the area of these polygons can be used to weight the station's rainfall data.
- **`basin_mean_func`**: This is the main function for calculating the basin's mean rainfall. It can perform either a simple arithmetic average or a weighted average using the weights derived from the Thiessen polygons.

### Example Usage

```python
import geopandas as gpd
import pandas as pd
from hydrodatasource.processor.basin_mean_rainfall import calculate_thiesen_polygons, basin_mean_func

# Load station locations and basin boundary
stations_gdf = gpd.read_file("path/to/stations.shp")
basin_gdf = gpd.read_file("path/to/basin.shp")

# Load rainfall data (as a DataFrame with station IDs as columns)
# Make sure the columns are sorted alphabetically
rainfall_df = pd.read_csv("path/to/rainfall.csv", index_col="time")
rainfall_df = rainfall_df.sort_index(axis=1)

# Calculate Thiessen polygons to get station weights
weights_gdf = calculate_thiesen_polygons(stations_gdf, basin_gdf)

# Create a weights dictionary
weights_dict = {
    tuple(weights_gdf["STCD"]): weights_gdf["area_ratio"].tolist()
}

# Calculate the basin mean rainfall
mean_rainfall = basin_mean_func(rainfall_df, weights_dict=weights_dict)

print(mean_rainfall)
```

## Rainfall-Runoff Event Identification (场次划分)

The `dmca_esr.py` module implements the DMCA-ESR method for identifying and separating individual rainfall-runoff events from continuous time series data. This is crucial for event-based hydrological modeling and analysis.

- **`get_rr_events`**: This is the primary function to use. It takes rainfall and streamflow data (as xarray DataArrays) and returns a dictionary where each key is a basin ID and the value is a pandas DataFrame listing the identified events.

### Example Usage

```python
import xarray as xr
from hydrodatasource.processor.dmca_esr import get_rr_events

# Assume 'rain_da' and 'flow_da' are xarray DataArrays with dimensions ('time', 'basin')
# and 'basin_area' is an xarray Dataset with the area for each basin.
rain_da = xr.open_dataset("path/to/rain.nc")["precipitation"]
flow_da = xr.open_dataset("path/to/flow.nc")["streamflow"]
basin_area = xr.open_dataset("path/to/attributes.nc")["area"]

# Identify rainfall-runoff events
rr_events = get_rr_events(rain_da, flow_da, basin_area)

# Print the events for a specific basin
for basin_id, events_df in rr_events.items():
    print(f"Events for basin {basin_id}:")
    print(events_df)
```
