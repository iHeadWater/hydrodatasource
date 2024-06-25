"""
Author: Wenyu Ouyang
Date: 2023-11-01 08:58:50
LastEditTime: 2024-06-25 09:42:12
LastEditors: Wenyu Ouyang
Description: Test funcs for reader.py
FilePath: \hydrodatasource\tests\test_reader.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import os
from minio import Minio
import hydrodataset as hds

from hydrodatasource.configs.config import LOCAL_DATA_PATH
from hydrodatasource.reader.reader import (
    AOI,
    DataHandler,
    LocalFileReader,
    MinioFileReader,
)


def test_gpm_read_minio():
    data_handler = DataHandler(
        aoi_type="basin",  # only basin for now, the streamflow data needs to be reorganized in the minio server,  then to be read.
        aoi_param="86_21401550",
        dataname="gpm",
        minio_read=True,
    )
    data = data_handler.handle()
    assert data is not None
    return data


def test_gfs_read_minio():
    data_handler = DataHandler(
        aoi_type="basin", aoi_param="86_21401550", dataname="gfs", minio_read=True
    )
    data = data_handler.handle()
    assert data is not None
    return data


def test_gpm_gfs_read_minio():
    data_handler = DataHandler(
        aoi_type="basin", aoi_param="86_21401550", dataname="gpm_gfs", minio_read=True
    )
    data = data_handler.handle()
    assert data is not None
    return data


def test_gpm_read_with_process():
    # if there is no data in minio, we want to merge a new one and save in local_data_path
    data_handler = DataHandler(
        aoi_type="basin",
        aoi_param="86_21401550",
        region="wis",
        time_periods=[["2017-01-01T00:00:00", "2017-01-31T00:00:00"]],
        dataname="gpm",
        minio_read=False,
        local_save=True,
    )
    data = data_handler.handle()
    assert data is not None
    return data


# The new method no longer needs ways below, but I keep them in case we need it.

# def test_reader_interface(minio_paras):
#     # 初始化Minio客户端
#     minio_server = minio_paras["endpoint_url"]
#     minio_client = Minio(
#         minio_server.replace("http://", ""),
#         access_key=minio_paras["access_key"],
#         secret_key=minio_paras["secret_key"],
#         secure=False,
#     )

#     gpm_handler = GPMDataHandler()
#     gfs_handler = GFSDataHandler()
#     aoi = AOI("grid", {"lat": 0, "lon": 0, "size": 1})

#     local_gpm_reader = LocalFileReader(gpm_handler)
#     local_gfs_reader = LocalFileReader(gfs_handler)
#     local_gpm_reader.read("path/to/file", aoi)

#     # Assume you have initialized the minio_client somewhere
#     minio_gpm_reader = MinioFileReader(minio_client, gpm_handler)
#     minio_gfs_reader = MinioFileReader(minio_client, gfs_handler)
#     minio_gpm_reader.read("path/to/file", aoi)


# def test_reader_station():
#     station_handler = StationDataHandler()
#     aoi = AOI(
#         "station",
#         {"station_id": "2181200", "start_time": "1980-01-01", "end_time": "2001-01-01"},
#     )

#     local_station_reader = LocalFileReader(station_handler)
#     data = local_station_reader.read(
#         os.path.join(LOCAL_DATA_PATH, "station.nc"),
#         aoi,
#     )
#     print(data)
