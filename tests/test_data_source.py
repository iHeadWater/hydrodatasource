"""
Author: Wenyu Ouyang
Date: 2024-07-06 19:20:59
LastEditTime: 2024-11-05 08:56:42
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
from hydrodatasource.configs.config import SETTING
from hydrodatasource.reader.data_source import CACHE_DIR, SelfMadeHydroDataset


@pytest.fixture
def one_hour_dataset():
    # local
    selfmadehydrodataset_path = SETTING["local_data_path"]["datasets-interim"]
    # minio
    # selfmadehydrodataset_path = "s3://basins-interim"
    return SelfMadeHydroDataset(data_path=selfmadehydrodataset_path, time_unit=["1h"])


@pytest.fixture
def three_hour_dataset():
    # local
    selfmadehydrodataset_path = SETTING["local_data_path"]["datasets-interim"]
    # minio
    # selfmadehydrodataset_path = "s3://basins-interim"
    return SelfMadeHydroDataset(data_path=selfmadehydrodataset_path, time_unit=["3h"])


@pytest.fixture
def one_day_dataset():
    # local
    selfmadehydrodataset_path = SETTING["local_data_path"]["datasets-interim"]
    # minio
    # selfmadehydrodataset_path = "s3://basins-interim"
    return SelfMadeHydroDataset(data_path=selfmadehydrodataset_path)


@pytest.fixture
def eight_day_dataset():
    # local
    selfmadehydrodataset_path = SETTING["local_data_path"]["datasets-interim"]
    # minio
    # selfmadehydrodataset_path = "s3://basins-interim"
    return SelfMadeHydroDataset(data_path=selfmadehydrodataset_path, time_unit=["8D"])


def test_selfmadehydrodataset_get_name(one_day_dataset):
    assert one_day_dataset.get_name() == "SelfMadeHydroDataset"


def test_selfmadehydrodataset_streamflow_unit(one_day_dataset):
    assert one_day_dataset.streamflow_unit == {"1D": "mm/d"}


def test_selfmadehydrodataset_read_site_info(one_day_dataset):
    site_info = one_day_dataset.read_site_info()
    assert isinstance(site_info, pd.DataFrame)


def test_selfmadehydrodataset_read_object_ids(one_day_dataset):
    object_ids = one_day_dataset.read_object_ids()
    assert isinstance(object_ids, np.ndarray)


def test_selfmadehydrodataset_read_tsdata(one_day_dataset):
    object_ids = one_day_dataset.read_object_ids()
    target_cols = one_day_dataset.read_timeseries(
        object_ids=object_ids[:5],
        t_range_list=["2020-01-01", "2020-12-31"],
        relevant_cols=["streamflow"],
        time_unit=["1D"],
    )
    assert isinstance(target_cols, dict)


def test_selfmadehydrodataset_read_attrdata(one_day_dataset):
    object_ids = one_day_dataset.read_object_ids()
    constant_cols = one_day_dataset.read_attributes(
        object_ids=object_ids[:5], constant_cols=["area"]
    )
    assert isinstance(constant_cols, np.ndarray)


def test_selfmadehydrodataset_get_attributes_cols(one_day_dataset):
    constant_cols = one_day_dataset.get_attributes_cols()
    assert isinstance(constant_cols, np.ndarray)


def test_selfmadehydrodataset_get_timeseries_cols(one_day_dataset):
    relevant_cols = one_day_dataset.get_timeseries_cols()
    assert isinstance(relevant_cols, dict)


def test_selfmadehydrodataset_cache_attributes_xrdataset(one_day_dataset):
    one_day_dataset.cache_attributes_xrdataset()
    assert os.path.exists(os.path.join(CACHE_DIR, "attributes.nc"))


def test_selfmadehydrodataset_cache_timeseries_xrdataset(
    one_day_dataset, three_hour_dataset, one_hour_dataset, eight_day_dataset
):
    # 8D
    eight_day_dataset.cache_timeseries_xrdataset(
        time_units=["8D"],
        t_range=["1980-01-01", "2023-12-31"],
        start0101_freq=True,
        batchsize=200,
    )
    # 1h
    one_hour_dataset.cache_timeseries_xrdataset(
        time_units=["1h"],
        t_range=["1980-01-01", "2023-12-31"],
    )
    # 3h
    three_hour_dataset.cache_timeseries_xrdataset(
        time_units=["3h"],
        t_range=["1980-01-01 01", "2023-12-31 22"],
    )
    # 1D
    one_day_dataset.cache_timeseries_xrdataset()


def test_selfmadehydrodataset_cache_xrdataset(one_day_dataset):
    one_day_dataset.cache_xrdataset()


def test_selfmadehydrodataset_read_ts_xrdataset(
    one_day_dataset, three_hour_dataset, one_hour_dataset, eight_day_dataset
):
    # 8D
    xrdataset_dict = eight_day_dataset.read_ts_xrdataset(
        gage_id_lst=["camels_01013500", "camels_01022500"],
        t_range=["2020-01-01", "2020-12-31"],
        var_lst=["ET_modis16a2006", "ET_modis16a2gf061"],
        time_units=["8D"],
    )
    target_cols = one_day_dataset.read_timeseries(
        object_ids=["camels_01013500", "camels_01022500"],
        t_range_list=["2020-01-01", "2020-12-31"],
        relevant_cols=["streamflow"],
        time_unit=["1D"],
    )
    assert isinstance(xrdataset_dict, dict)
    np.testing.assert_array_equal(
        xrdataset_dict["1D"]["streamflow"].values, target_cols["1D"][:, :, 0]
    )
    # 1h
    xrdataset_dict = one_hour_dataset.read_ts_xrdataset(
        gage_id_lst=["camels_01013500", "camels_01022500"],
        t_range=["2020-01-01", "2020-12-31 23"],
        var_lst=["streamflow"],
        time_units=["1h"],
    )
    target_cols = one_hour_dataset.read_timeseries(
        object_ids=["camels_01013500", "camels_01022500"],
        t_range_list=["2020-01-01", "2020-12-31 23"],
        relevant_cols=["streamflow"],
        time_units=["1h"],
    )
    assert isinstance(xrdataset_dict, dict)
    np.testing.assert_array_equal(
        xrdataset_dict["1h"]["streamflow"].values, target_cols["1h"][:, :, 0]
    )

    # 3h
    xrdataset_dict = three_hour_dataset.read_ts_xrdataset(
        gage_id_lst=["camels_01013500", "camels_01022500"],
        t_range=["2020-01-01 01", "2020-12-31 22"],
        var_lst=["streamflow"],
        time_units=["3h"],
    )
    target_cols = three_hour_dataset.read_timeseries(
        object_ids=["camels_01013500", "camels_01022500"],
        t_range_list=["2020-01-01 01", "2020-12-31 22"],
        relevant_cols=["streamflow"],
        time_units=["3h"],
    )
    assert isinstance(xrdataset_dict, dict)
    np.testing.assert_array_equal(
        xrdataset_dict["3h"]["streamflow"].values, target_cols["3h"][:, :, 0]
    )

    # 1D
    xrdataset_dict = one_day_dataset.read_ts_xrdataset(
        gage_id_lst=["camels_01013500", "camels_01022500"],
        t_range=["2020-01-01", "2020-12-31"],
        var_lst=["streamflow"],
        time_units=["1D"],
    )
    target_cols = one_day_dataset.read_timeseries(
        object_ids=["camels_01013500", "camels_01022500"],
        t_range_list=["2020-01-01", "2020-12-31"],
        relevant_cols=["streamflow"],
        time_unit=["1D"],
    )
    assert isinstance(xrdataset_dict, dict)
    np.testing.assert_array_equal(
        xrdataset_dict["1D"]["streamflow"].values, target_cols["1D"][:, :, 0]
    )


def test_selfmadehydrodataset_read_attr_xrdataset(one_day_dataset):
    xrdataset = one_day_dataset.read_attr_xrdataset(
        gage_id_lst=["camels_01013500", "camels_01022500"],
        var_lst=["area"],
    )
    assert isinstance(xrdataset, xr.Dataset)


def test_selfmadehydrodataset_read_area(one_day_dataset):
    area = one_day_dataset.read_area(gage_id_lst=["camels_01013500", "camels_01022500"])
    assert isinstance(area, xr.Dataset)


def test_selfmadehydrodataset_read_mean_prcp(one_day_dataset):
    mean_prcp = one_day_dataset.read_mean_prcp(
        gage_id_lst=["camels_01013500", "camels_01022500"]
    )
    assert isinstance(mean_prcp, xr.Dataset)
    assert mean_prcp["pre_mm_syr"].attrs["units"] == "mm/d"


def test_read_mean_prcp_mm_per_hour(one_day_dataset):
    mean_prcp = one_day_dataset.read_mean_prcp(
        gage_id_lst=["camels_01013500", "camels_01022500"], unit="mm/h"
    )
    mean_prcp_ = one_day_dataset.read_mean_prcp(
        gage_id_lst=["camels_01013500", "camels_01022500"]
    )
    assert isinstance(mean_prcp, xr.Dataset)
    assert mean_prcp["pre_mm_syr"].attrs["units"] == "mm/h"
    np.testing.assert_allclose(
        mean_prcp["pre_mm_syr"].values, mean_prcp_["pre_mm_syr"].values / 24
    )


def test_read_mean_prcp_mm_per_3hour(one_day_dataset):
    mean_prcp = one_day_dataset.read_mean_prcp(
        gage_id_lst=["camels_01013500", "camels_01022500"], unit="mm/3h"
    )
    mean_prcp_ = one_day_dataset.read_mean_prcp(
        gage_id_lst=["camels_01013500", "camels_01022500"]
    )
    assert isinstance(mean_prcp, xr.Dataset)
    assert mean_prcp["pre_mm_syr"].attrs["units"] == "mm/3h"
    np.testing.assert_allclose(
        mean_prcp["pre_mm_syr"].values, mean_prcp_["pre_mm_syr"].values / (24 / 3)
    )


def test_read_mean_prcp_mm_per_8day(one_day_dataset):
    mean_prcp = one_day_dataset.read_mean_prcp(
        gage_id_lst=["camels_01013500", "camels_01022500"], unit="mm/8d"
    )
    mean_prcp_ = one_day_dataset.read_mean_prcp(
        gage_id_lst=["camels_01013500", "camels_01022500"]
    )
    assert isinstance(mean_prcp, xr.Dataset)
    assert mean_prcp["pre_mm_syr"].attrs["units"] == "mm/8d"
    np.testing.assert_allclose(
        mean_prcp["pre_mm_syr"].values, mean_prcp_["pre_mm_syr"].values * 8
    )


def test_read_mean_prcp_invalid_unit(one_day_dataset):
    with pytest.raises(ValueError, match="unit must be one of"):
        one_day_dataset.read_mean_prcp(
            gage_id_lst=["camels_01013500", "camels_01022500"], unit="invalid_unit"
        )
