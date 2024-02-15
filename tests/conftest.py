"""

from hydrodata.common import minio_cfg
Author: Wenyu Ouyang
Date: 2023-10-31 09:41:59
LastEditTime: 2024-02-13 17:34:48
LastEditors: Wenyu Ouyang
Description: conf for pytest
FilePath: \hydrodata\tests\conftest.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""
import pytest

from hydrodata.configs.common import minio_cfg


@pytest.fixture()
def minio_paras():
    return minio_cfg(bucket_name="test-private-data")
