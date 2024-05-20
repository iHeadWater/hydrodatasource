"""
Author: Ynag Wang
Date: 2023-11-02 14:52:08
LastEditTime: 2024-03-28 08:39:39
LastEditors: Wenyu Ouyang
Description: 
FilePath: \hydrodatasource\tests\test_sync.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import os

import pytest

from hydrodatasource.configs import config
from hydrodatasource.reader.minio_api import minio_sync_files, boto3_sync_files

pytest_plugins = ("pytest_asyncio",)


@pytest.mark.asyncio
async def test_sync_data():
    s3_client = config.S3
    mc_client = config.MC
    await minio_sync_files(
        mc_client,
        "forestbat-private",
        local_path=os.path.join(config.LOCAL_DATA_PATH, "forestbat_test"),
    )
    await boto3_sync_files(
        s3_client,
        "forestbat-private",
        local_path=os.path.join(config.LOCAL_DATA_PATH, "forestbat_test_1"),
    )
