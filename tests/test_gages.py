"""
Author: Wenyu Ouyang
Date: 2024-11-04 19:50:06
LastEditTime: 2025-11-03 17:14:00
LastEditors: Wenyu Ouyang
Description: test for gages
FilePath: \hydrodatasource\tests\test_gages.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import os
import pytest
import pandas as pd
from hydrodatasource.reader.gages import Gages
from hydrodatasource.configs.config import CACHE_DIR, SETTING


@pytest.fixture
def gages_dataset():
    # local
    gages_path = os.path.join(SETTING["local_data_path"]["datasets-interim"])
    # minio
    # gages_path = "s3://basins-interim"
    return Gages(data_path=gages_path, dataset_name="hydrodl-reservoir-jh-paper")


def test_gages_read_site_info(gages_dataset):
    site_info = gages_dataset.read_site_info()
    assert isinstance(site_info, pd.DataFrame)


def test_gages_cache_attributes_xrdataset(gages_dataset):
    gages_dataset.cache_attributes_xrdataset()
    assert os.path.exists(
        os.path.join(CACHE_DIR, "hydrodl-reservoir-jh-paper_attributes.nc")
    )
