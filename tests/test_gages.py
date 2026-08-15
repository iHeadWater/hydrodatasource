"""
Author: Wenyu Ouyang
Date: 2024-11-04 19:50:06
LastEditTime: 2025-11-04 10:16:31
LastEditors: Wenyu Ouyang
Description: test for gages
FilePath: \hydrodatasource\tests\test_gages.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import os
import pytest
import pandas as pd
from hydrodatasource.reader.gages import Gages
from hydrodatasource.configs.config import CACHE_DIR

pytestmark = pytest.mark.internal_data

_GAGES_DATA = os.environ.get("HDS_GAGES_DATA")


@pytest.fixture
def gages_dataset():
    if not _GAGES_DATA:
        pytest.skip("Set HDS_GAGES_DATA env var")
    return Gages(_GAGES_DATA)


def test_gages_read_site_info(gages_dataset):
    site_info = gages_dataset.read_site_info()
    assert isinstance(site_info, pd.DataFrame)


def test_gages_cache_attributes_xrdataset(gages_dataset):
    gages_dataset.cache_attributes_xrdataset()
    assert os.path.exists(
        os.path.join(CACHE_DIR, "hydrodl-reservoir-jh-paper_attributes.nc")
    )
