"""
Author: Shuolong Xu
Date: 2024-02-15 16:40:34
LastEditTime: 2024-03-28 08:42:05
LastEditors: Wenyu Ouyang
Description: Test cases for gpm and gfs data
FilePath: \hydrodatasource\tests\test_gpm_gfs.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import os
from datetime import datetime

from hydrodatasource.processor.mask import gen_single_mask
from hydrodatasource.configs.config import LOCAL_DATA_PATH
from hydrodatasource.processor.gpm_gfs import make1nc41basin
from hydrodatasource.utils.utils import generate_time_intervals


def test_gen_mask():
    mask = gen_single_mask(
        basin_id="1_02051500",
        dataname="gpm",
    )
    assert mask is not None
    return mask


def test_gen_mask_minio():
    mask = gen_single_mask(
        basin_id="10310500",
        dataname="gpm",
        mask_path=os.path.join(LOCAL_DATA_PATH, "datasets-origin", "mask"),
        minio=True,
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
        mask_path=os.path.join(LOCAL_DATA_PATH, "datasets-origin", "mask"),
        shp_path=os.path.join(LOCAL_DATA_PATH, "datasets-origin", "shp"),
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
        mask_path=os.path.join(LOCAL_DATA_PATH, "datasets-origin", "mask"),
        shp_path=os.path.join(LOCAL_DATA_PATH, "datasets-origin", "shp"),
        dataset="camels",
        time_periods=[["2017-01-01T00:00:00", "2017-01-31T00:00:00"]],
        local_save=True,
        minio_upload=False,
    )
    assert data is not None
    return data


def test_merge_without_local_data():
    data = make1nc41basin(
        basin_id="1_02051500",
        dataname="merge",
        local_path=LOCAL_DATA_PATH,
        mask_path=os.path.join(LOCAL_DATA_PATH, "datasets-origin", "mask"),
        shp_path=os.path.join(LOCAL_DATA_PATH, "datasets-origin", "shp"),
        dataset="camels",
        time_periods=[
            ["2017-01-10T00:00:00", "2017-01-11T00:00:00"],
            ["2017-01-15T00:00:00", "2017-01-20T00:00:00"],
        ],
        local_save=True,
        minio_upload=False,
        gpm_length=169,
        gfs_length=23,
        time_now_length=168,
    )
    assert data is not None
    return data


def test_merge_with_local_data():
    data = make1nc41basin(
        basin_id="1_02051500",
        dataname="merge",
        local_path=LOCAL_DATA_PATH,
        mask_path=os.path.join(LOCAL_DATA_PATH, "datasets-origin", "mask"),
        shp_path=os.path.join(LOCAL_DATA_PATH, "datasets-origin", "shp"),
        dataset="camels",
        time_periods=[["2017-01-01T00:00:00", "2017-01-31T00:00:00"]],
        local_save=True,
        minio_upload=False,
        gpm_length=169,
        gfs_length=23,
        time_now_length=168,
        gpm_path=os.path.join(
            LOCAL_DATA_PATH, "datasets-interim", "1_02051500", "gpm.nc"
        ),
        gfs_path=os.path.join(
            LOCAL_DATA_PATH, "datasets-interim", "1_02051500", "gfs.nc"
        ),
    )
    assert data is not None
    return data
