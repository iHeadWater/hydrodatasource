#!/usr/bin/env python
"""
Author: Wenyu Ouyang
Date: 2023-10-25 15:16:21
LastEditTime: 2023-10-26 08:56:11
LastEditors: Wenyu Ouyang
Description: Tests for preprocess
FilePath: \hydro_privatedata\tests\test_hydroprivatedata.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""
import os.path
import pathlib

from hydroprivatedata import preprocess, minio_api, config
from hydroprivatedata.preprocess import huanren_preprocess


def test_content():
    preprocess.huanren_preprocess()


def test_upload_csv():
    test_path = pathlib.Path(os.path.abspath(os.path.curdir)).parent
    client = config.mc
    bucket_name = config.minio_paras['bucket_path']
    minio_api.boto3_upload_csv(client, bucket_name, 'nyc_taxi', os.path.join(test_path, 'test_data/nyc_taxi.csv'))
    minio_api.minio_upload_csv(client, bucket_name, 'driver_data_site24', os.path.join(test_path, 'test_data/driver_data_site24.csv'))


def test_huanren_preprocess():
    huanren_preprocess()
