"""
Author: Wenyu Ouyang
Date: 2023-01-02 22:23:24
LastEditTime: 2024-03-28 08:34:57
LastEditors: Wenyu Ouyang
Description: read the Global Runoff Data Centre (GRDC) daily data
FilePath: \hydrodatasource\hydrodatasource\reader\grdc.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""
# Global Runoff Data Centre module from ewatercycle: https://github.com/eWaterCycle/ewatercycle/blob/main/src/ewatercycle/observation/grdc.py
import datetime
import os
import pandas as pd
import xarray as xr

from hydrodatasource.downloader.hydrostation import catalogue_grdc
from hydrodatasource.processor.grdc import read_grdc_daily_data


def dailygrdc2netcdf(start_date, end_date, data_dir=None, station_ids=None):
    """
    Parameters
    ----------
    start_date : _type_
        a startDate provided in YYYY-MM-DD
    end_date : _type_
        a endDate provided in YYYY-MM-DD
    """
    nc_file = os.path.join(data_dir, "grdc_daily_data.nc")
    if os.path.exists(nc_file):
        return

    catalogue = catalogue_grdc(data_dir)
    # Create empty lists to store data and metadata
    data_list = []
    meta_list = []

    if station_ids is None:
        # Filter the catalogue based on the provided station IDs
        filenames = os.listdir(data_dir)
        # Extract station IDs from filenames that match the pattern
        station_ids = [
            int(fname.split("_")[0])
            for fname in filenames
            if fname.endswith("_Q_Day.Cmd.txt")
        ]
    catalogue = catalogue[catalogue["grdc_no"].isin(station_ids)]

    # Loop over each station in the catalogue
    for station_id in catalogue["grdc_no"]:
        try:
            st = datetime.datetime.strptime(start_date, "%Y-%m-%d").strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )
            et = datetime.datetime.strptime(end_date, "%Y-%m-%d").strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )
            df, meta = read_grdc_daily_data(str(station_id), st, et, data_dir)
        except Exception as e:
            print(f"Error reading data for station {station_id}: {e}")
            # Create an empty DataFrame with the same structure
            df = pd.DataFrame(
                columns=["streamflow"],
                index=pd.date_range(start=start_date, end=end_date),
            )
            df["streamflow"] = float("nan")
            meta = {
                "grdc_file_name": "",
                "id_from_grdc": station_id,
                "river_name": "",
                "station_name": "",
                "country_code": "",
                "grdc_latitude_in_arc_degree": float("nan"),
                "grdc_longitude_in_arc_degree": float("nan"),
                "grdc_catchment_area_in_km2": float("nan"),
                "altitude_masl": float("nan"),
                "dataSetContent": "",
                "units": "m³/s",
                "time_series": "",
                "no_of_years": 0,
                "last_update": "",
                "nrMeasurements": "NA",
                "UserStartTime": start_date,
                "UserEndTime": end_date,
                "nrMissingData": 0,
            }

        # Convert the DataFrame to an xarray DataArray and append to the list
        # da = xr.DataArray(
        #     df["streamflow"].values,
        #     coords=[df.index, [station_id]],
        #     dims=["time", "station"],
        # )
        coords_dict = {"time": df.index, "station": [station_id]}
        da = xr.DataArray(
            data=df["streamflow"].values.reshape(-1, 1),
            coords=coords_dict,
            dims=["time", "station"],
            name="streamflow",
            attrs={"units": meta.get("units", "unknown")},
        )
        data_list.append(da)

        # Append metadata
        meta_list.append(meta)

    # Concatenate all DataArrays along the 'station' dimension
    ds = xr.concat(data_list, dim="station")

    # Assign attributes
    ds.attrs["description"] = "Daily river discharge"
    ds.station.attrs["description"] = "GRDC station number"

    # Write the xarray Dataset to a NetCDF file
    ds.to_netcdf(nc_file)

    print("NetCDF file created successfully!")


class GRDCDataHandler:
    def handle(self, configuration):
        aoi_param = configuration["aoi"].aoi_param
        start_time = aoi_param["start_time"]
        end_time = aoi_param["end_time"]
        station_id = aoi_param["station_id"]
        # Based on configuration, read and handle GRDC data specifically
        nc_file = configuration["path"]
        if not os.path.isfile(nc_file):
            dailygrdc2netcdf(start_time, end_time, data_dir=os.path.dirname(nc_file))
        ds = xr.open_dataset(nc_file)
        # choose data for given basin
        return ds.sel(station=int(station_id)).sel(time=slice(start_time, end_time))
