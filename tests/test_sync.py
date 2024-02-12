"""
Author: Wenyu Ouyang
Date: 2023-11-02 14:52:08
LastEditTime: 2024-02-12 15:37:00
LastEditors: Wenyu Ouyang
Description: 
FilePath: \hydrodata\tests\test_sync.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""
import os

import pytest

from hydrodata import config
from hydrodata.minio_api import minio_sync_files, boto3_sync_files

pytest_plugins = ('pytest_asyncio',)


@pytest.mark.asyncio
async def test_sync_data():
    s3_client = config.s3
    mc_client = config.mc
    await minio_sync_files(mc_client, 'forestbat-private',
                           local_path=os.path.join(config.LOCAL_DATA_PATH, 'forestbat_test'))
    await boto3_sync_files(s3_client, 'forestbat-private',
                           local_path=os.path.join(config.LOCAL_DATA_PATH, 'forestbat_test_1'))
