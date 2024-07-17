"""
Author: Wenyu Ouyang
Date: 2024-07-06 19:20:59
LastEditTime: 2024-07-17 16:24:14
LastEditors: Wenyu Ouyang
Description: Test funcs for data source
FilePath: \hydrodatasource\tests\test_data_source.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import os
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from hydrodatasource.reader.data_source import CACHE_DIR, SelfMadeHydroDataset


@pytest.fixture
def dataset():
    own_datapath = "C:\\Users\\wenyu\\OneDrive\\Research\\research_topic_advancement\\research_progress_plan\\data4dpl\\dplARdata"
    return SelfMadeHydroDataset(data_path=own_datapath)


def test_selfmadehydrodataset_get_name(dataset):
    assert dataset.get_name() == "SelfMadeHydroDataset"


def test_selfmadehydrodataset_streamflow_unit(dataset):
    assert dataset.streamflow_unit == "m^3/s"


def test_selfmadehydrodataset_read_site_info(dataset):
    site_info = dataset.read_site_info()
    assert isinstance(site_info, pd.DataFrame)


def test_selfmadehydrodataset_read_object_ids(dataset):
    object_ids = dataset.read_object_ids()
    assert isinstance(object_ids, np.ndarray)


def test_selfmadehydrodataset_read_tsdata(dataset):
    object_ids = dataset.read_object_ids()
    target_cols = dataset.read_timeseries(
        object_ids=object_ids[:5],
        t_range_list=["2020-01-01", "2020-12-31"],
        relevant_cols=["streamflow"],
    )
    assert isinstance(target_cols, np.ndarray)


def test_selfmadehydrodataset_read_attrdata(dataset):
    object_ids = dataset.read_object_ids()
    constant_cols = dataset.read_attributes(
        object_ids=object_ids[:5], constant_cols=["area"]
    )
    assert isinstance(constant_cols, np.ndarray)


def test_selfmadehydrodataset_get_attributes_cols(dataset):
    constant_cols = dataset.get_attributes_cols()
    assert isinstance(constant_cols, np.ndarray)


def test_selfmadehydrodataset_get_timeseries_cols(dataset):
    relevant_cols = dataset.get_timeseries_cols()
    assert isinstance(relevant_cols, np.ndarray)


def test_selfmadehydrodataset_cache_attributes_xrdataset(dataset):
    dataset.cache_attributes_xrdataset(region="dPL")
    assert os.path.exists(os.path.join(CACHE_DIR, "dPL_attributes.nc"))


def test_selfmadehydrodataset_cache_timeseries_xrdataset(dataset):
    dataset.cache_timeseries_xrdataset(region="dPL")


def test_selfmadehydrodataset_cache_xrdataset(dataset):
    dataset.cache_xrdataset()


def test_selfmadehydrodataset_read_ts_xrdataset(dataset):
    xrdataset = dataset.read_ts_xrdataset(
        gage_id_lst=["camels_01013500", "camels_01022500"],
        t_range=["2020-01-01", "2020-12-31"],
        var_lst=["streamflow"],
        region="dPL",
    )
    assert isinstance(xrdataset, xr.Dataset)


def test_selfmadehydrodataset_read_attr_xrdataset(dataset):
    xrdataset = dataset.read_attr_xrdataset(
        gage_id_lst=["camels_01013500", "camels_01022500"],
        var_lst=["area"],
        region="dPL",
    )
    assert isinstance(xrdataset, xr.Dataset)


def test_selfmadehydrodataset_read_area(dataset):
    area = dataset.read_area(gage_id_lst=["camels_01013500", "camels_01022500"])
    assert isinstance(area, xr.Dataset)


def test_selfmadehydrodataset_read_mean_prcp(dataset):
    mean_prcp = dataset.read_mean_prcp(
        gage_id_lst=["camels_01013500", "camels_01022500"]
    )
    assert isinstance(mean_prcp, xr.Dataset)
