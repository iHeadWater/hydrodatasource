"""
Author: Jiaxu Wu
Date: 2024-01-12 15:21:20
LastEditTime: 2024-03-28 08:37:54
LastEditors: Wenyu Ouyang
Description: processor for era5 data
FilePath: \hydrodatasource\hydrodatasource\processor\era5.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""
import pandas as pd
import geopandas as gpd
import xarray as xr

from hydro_gistools.mean import gen_mask, mean_by_mask

from hydrodatasource.reader.minio import ERA5LReader

def get_era5():
    era5_land = ERA5LReader()

    data_variables = ["Total precipitation", "2 metre temperature"]
    start_time = pd.to_datetime("2016-01-01")
    end_time = pd.to_datetime("2020-12-31")
    shp = "/ftproot/LSTM_data/碧流河流域.shp"
    save_file = "data/era5_land_5.nc"

    era5_land.to_netcdf(
        data_variables=data_variables,
        start_time=start_time,
        end_time=end_time,
        shp=shp,
        save_file=save_file,
    )


def get_mean():
    watershed = gpd.read_file("/ftproot/LSTM_data/碧流河流域.shp")
    # hydro_gistools mean.py line 249 overlay() keep_geom_type=False without warning
    gen_mask(watershed, "FID", "gfs", save_dir="data/mask")
    gen_mask(watershed, "FID", "gpm", save_dir="data/mask")
    gen_mask(watershed, "FID", "era5", save_dir="data/mask")
    data = xr.open_dataset("/ftproot/LSTM_data/era5_land_5.nc")
    mask = xr.open_dataset("data/mask/mask-1-era5.nc")
    mean_tp = mean_by_mask(data, "tp", mask)
    print("mean_tp computed!")
    mean_t2m = mean_by_mask(data, "t2m", mask)
    print("mean_t2m computed!")
    df = pd.DataFrame()
    df["TM"] = data["time"]
    df["tp"] = mean_tp.compute()
    df["t2m"] = mean_t2m.compute()
    df.to_csv("data/era5_mean.csv", index=False)
    print("data/era5_mean.csv saved!")
    basin_value = "21401550"
    ds = xr.Dataset(
        {
            "prcp": (("time", "basin"), df[["tp"]].values),
            "temperature": (("time", "basin"), df[["t2m"]].values),
        },
        coords={"time": df["TM"].values, "basin": [basin_value]},
    )
    ds["time"] = ds["time"].astype("datetime64[ns]")
    nc_file_path = "data/era5_mean.nc"
    ds.to_netcdf(nc_file_path)
    print("data/era5_mean.nc saved!")


get_mean()
