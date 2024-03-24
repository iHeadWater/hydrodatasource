#!/usr/bin/env python
"""
Author: Wenyu Ouyang
Date: 2023-10-25 15:16:21
LastEditTime: 2024-03-24 09:24:59
LastEditors: Wenyu Ouyang
Description: Tests for reading basins' data
FilePath: \hydrodata\tests\test_reader_basins.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""
import os.path
import pathlib

from hydrodata.reader import stations
from hydrodata.configs import config
from hydrodata.reader import minio_api


def test_content():
    # preprocess.huanren_preprocess()
    stations.biliu_stbprp_decode()


def test_upload_csv():
    test_path = pathlib.Path(os.path.abspath(os.path.curdir)).parent
    client = config.MC
    bucket_name = config.MINIO_PARAM["bucket_path"]
    minio_api.boto3_upload_csv(
        client,
        bucket_name,
        "nyc_taxi",
        os.path.join(test_path, "test_data/nyc_taxi.csv"),
    )
    minio_api.minio_upload_csv(
        client,
        bucket_name,
        "driver_data_site24",
        os.path.join(test_path, "test_data/driver_data_site24.csv"),
    )


def test_download_csv_minio():
    client = config.S3
    bucket_name = config.MINIO_PARAM["bucket_path"]
    # minio_api.minio_download_csv(client, bucket_name, 'nyc_taxi', file_path='test_dload')
    minio_api.boto3_download_csv(
        client, bucket_name, "driver_data_site24", "driver_data_site24.csv"
    )
