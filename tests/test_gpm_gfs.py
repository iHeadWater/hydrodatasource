"""
Author: Shuolong Xu
Date: 2024-02-15 16:40:34
LastEditTime: 2024-02-20 19:55:03
LastEditors: Wenyu Ouyang
Description: Test cases for gpm and gfs data
FilePath: \hydrodata\tests\test_gpm_gfs.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""
from hydrodata.processor.mask import gen_single_mask
from hydrodata.configs.config import LOCAL_DATA_PATH
from hydrodata.processor.gpm_gfs import make1nc41basin
from hydrodata.utils.utils import generate_time_intervals
import os
from datetime import datetime


def test_gen_mask():
    mask_path = os.path.join(LOCAL_DATA_PATH, "data_origin", "mask")
    mask = gen_single_mask(
        basin_id="1_02051500",
        shp_path=os.path.join(LOCAL_DATA_PATH, "data_origin", "shp"),
        dataname="gfs",
        mask_path=mask_path,
    )
    assert mask is not None
    return mask


def test_time_intervals():
    time = generate_time_intervals(
        start_date=datetime(2017, 1, 1), end_date=datetime(2017, 1, 3)
    )
    assert time is not None
    return time


def test_gpm():
    data = make1nc41basin(
        basin_id="1_02051500",
        dataname="gpm",
        local_path=LOCAL_DATA_PATH,
        mask_path=os.path.join(LOCAL_DATA_PATH, "data_origin", "mask"),
        shp_path=os.path.join(LOCAL_DATA_PATH, "data_origin", "shp"),
        dataset="camels",
        time_periods=[["2017-01-01T00:00:00", "2017-01-31T00:00:00"]],
        local_save=True,
        minio_upload=False,
    )
    assert data is not None
    return data


def test_gfs():
    data = make1nc41basin(
        basin_id="1_02051500",
        dataname="gfs",
        local_path=LOCAL_DATA_PATH,
        mask_path=os.path.join(LOCAL_DATA_PATH, "data_origin", "mask"),
        shp_path=os.path.join(LOCAL_DATA_PATH, "data_origin", "shp"),
        dataset="camels",
        time_periods=[["2017-01-01T00:00:00", "2017-01-31T00:00:00"]],
        local_save=True,
        minio_upload=False,
    )
    assert data is not None
    return data


def test_merge():
    pass
