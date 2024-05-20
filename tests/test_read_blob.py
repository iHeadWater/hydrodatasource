"""
Author: Yang Wang
Date: 2024-03-16 15:55:22
LastEditTime: 2024-03-28 08:40:55
LastEditors: Wenyu Ouyang
Description: 
FilePath: \hydrodatasource\tests\test_read_blob.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""
import time
import intake
import ujson
import xarray as xr
import zarr
from distributed import LocalCluster
from kerchunk.hdf import SingleHdf5ToZarr

import hydrodatasource.configs.config as conf


# @pytest.mark.asyncio
def test_read_blob():
    """
    start_time = time.time()
    cluster = LocalCluster(dashboard_address=":10086")  # Fully-featured local Dask cluster
    client = cluster.get_client()
    # ds = xr.open_dataset('/home/forestbat/usgs-streamflow-nldas_hourly.nc')
    ds = xr.open_dataset('/home/forestbat/gpm_gfs.nc')
    # new_zarr = ds.to_zarr('usgs-streamflow-nldas_hourly.zarr', mode='w')
    print(time.time() - start_time)
    """
    """
    zarr_start_time = time.time()
    # https://pastebin.com/fKKECf3U
    zarr_path = conf.FS.get_mapper('s3://datasets-origin/usgs_streamflow_nldas_hourly.zarr')
    wrapped_store = zarr.storage.KVStore(zarr_path)
    zds = xr.open_zarr(wrapped_store)
    print(time.time() - zarr_start_time)
    ###########################
    cluster = LocalCluster(dashboard_address=":10086")  # Fully-featured local Dask cluster
    client = cluster.get_client()
    dask_zarr_start_time = time.time()
    zarr_dask_path = conf.FS.get_mapper('s3://datasets-origin/usgs_streamflow_nldas_hourly.zarr')
    zarr_dask_store = zarr.storage.KVStore(zarr_dask_path)
    dzds = xr.open_zarr(zarr_dask_store)
    print(time.time() - dask_zarr_start_time)
    """
    ###########################
    """
    # 5~7 min
    nc_to_zarr_time = time.time()
    nc_chunks = SingleHdf5ToZarr(h5f='/home/forestbat/usgs-streamflow-nldas_hourly.nc')
    with open('gpm_gfs_nc.json', 'wb') as fpj:
        fpj.write(ujson.dumps(nc_chunks.translate()).encode())
    print(time.time() - nc_to_zarr_time)
    """
    ###########################
    virtual_zarr_start_time = time.time()
    """
    # Method 0
    vds = xr.open_dataset("reference://", engine="zarr", backend_kwargs={
        "storage_options": {"fo": 'gpm_gfs_nc.json'},
        "remote_options": conf.MINIO_PARAM,
        "remote_protocol": "s3",
    })
    """
    # ~2mints
    vds = xr.open_dataset("gpm_gfs_nc.json", engine="kerchunk")
    print(time.time() - virtual_zarr_start_time)
