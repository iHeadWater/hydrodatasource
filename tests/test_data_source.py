"""
Author: Wenyu Ouyang
Date: 2024-07-06 19:20:59
LastEditTime: 2025-11-03 16:43:36
LastEditors: Wenyu Ouyang
Description: Test funcs for data source
FilePath: \hydrodatasource\tests\test_data_source.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import pytest

pytestmark = pytest.mark.internal_data
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from hydrodatasource.configs.config import CACHE_DIR, SETTING
from hydrodatasource.reader.data_source import (
    SelfMadeForecastDataset,
    SelfMadeHydroDataset,
)


@pytest.fixture
def one_hour_dataset():
    # local
    selfmadehydrodataset_path = SETTING["local_data_path"]["datasets-interim"]
    # minio
    # selfmadehydrodataset_path = "s3://basins-interim"
    return SelfMadeHydroDataset(
        data_path=selfmadehydrodataset_path, dataset_name="FDSources", time_unit=["1h"]
    )


@pytest.fixture
def three_hour_dataset():
    # local
    selfmadehydrodataset_path = SETTING["local_data_path"]["datasets-interim"]
    # minio
    # selfmadehydrodataset_path = "s3://basins-interim"
    return SelfMadeHydroDataset(
        data_path=selfmadehydrodataset_path, dataset_name="FDSources", time_unit=["3h"]
    )


@pytest.fixture
def one_day_dataset():
    # local
    selfmadehydrodataset_path = SETTING["local_data_path"]["datasets-interim"]
    # minio
    # selfmadehydrodataset_path = "s3://basins-interim"
    return SelfMadeHydroDataset(
        data_path=selfmadehydrodataset_path, dataset_name="FDSources"
    )


@pytest.fixture
def eight_day_dataset():
    # local
    selfmadehydrodataset_path = SETTING["local_data_path"]["datasets-interim"]
    # minio
    # selfmadehydrodataset_path = "s3://basins-interim"
    return SelfMadeHydroDataset(
        data_path=selfmadehydrodataset_path,
        dataset_name="FDSources",
        time_unit=["8D"],
    )


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


@pytest.mark.slow
def test_selfmadehydrodataset_cache_timeseries_xrdataset(
    one_day_dataset, three_hour_dataset, one_hour_dataset, eight_day_dataset
):
    # 8D
    eight_day_dataset.cache_timeseries_xrdataset(
        time_units=["8D"],
        trange4cache=["1980-01-01", "2023-12-31"],
        start0101_freq=True,
        batchsize=200,
    )
    # 1h
    one_hour_dataset.cache_timeseries_xrdataset(
        time_units=["1h"],
        trange4cache=["1980-01-01", "2023-12-31"],
    )
    # 3h
    three_hour_dataset.cache_timeseries_xrdataset(
        time_units=["3h"],
        trange4cache=["1980-01-01 01", "2023-12-31 22"],
    )
    # 1D
    one_day_dataset.cache_timeseries_xrdataset()


@pytest.mark.slow
def test_selfmadehydrodataset_cache_xrdataset(one_day_dataset):
    one_day_dataset.cache_xrdataset()


def test_selfmadehydrodataset_read_ts_xrdataset(
    one_day_dataset, three_hour_dataset, one_hour_dataset, eight_day_dataset
):
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
        t_range=["2020-01-01 00", "2020-12-31 00"],
        var_lst=["streamflow"],
        time_units=["3h"],
        start_hour_in_a_day=0,
    )
    target_cols = three_hour_dataset.read_timeseries(
        object_ids=["camels_01013500", "camels_01022500"],
        t_range_list=["2020-01-01 00", "2020-12-31 00"],
        relevant_cols=["streamflow"],
        time_units=["3h"],
        start_hour_in_a_day=0,
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

    # 8D
    xrdataset_dict = eight_day_dataset.read_ts_xrdataset(
        gage_id_lst=["camels_01013500", "camels_01022500"],
        t_range=["2020-01-01", "2020-12-31"],
        var_lst=["ET_modis16a2006", "ET_modis16a2gf061"],
        time_units=["8D"],
    )
    target_cols = eight_day_dataset.read_timeseries(
        object_ids=["camels_01013500", "camels_01022500"],
        t_range_list=["2020-01-01", "2020-12-31"],
        relevant_cols=["ET_modis16a2006", "ET_modis16a2gf061"],
        time_units=["8D"],
    )
    assert isinstance(xrdataset_dict, dict)
    np.testing.assert_array_equal(
        xrdataset_dict["8D"]["ET_modis16a2006"].values, target_cols["8D"][:, :, 0]
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


@pytest.fixture
def one_day_forecast_dataset():
    # local
    selfmadehydrodataset_path = SETTING["local_data_path"]["datasets-interim"]
    # minio
    # selfmadehydrodataset_path = "s3://basins-interim"
    return SelfMadeForecastDataset(
        data_path=selfmadehydrodataset_path, dataset_name="FDSources"
    )


def test_read_forecast_multiple_basins_all_exist(mocker, one_day_forecast_dataset):
    mocker.patch.object(
        one_day_forecast_dataset,
        "data_source_description",
        {"FORECAST_DIR": "/mock/forecast_dir"},
    )
    mocker.patch("os.path.exists", return_value=True)
    mock_data = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=3),
            "forecast_date": pd.date_range("2020-01-01", periods=3),
            "streamflow": [1.0, 2.0, 3.0],
        }
    )
    mocker.patch("pandas.read_csv", return_value=mock_data)
    result = one_day_forecast_dataset.read_forecast(
        object_ids=["basin_1", "basin_2"],
        t_range_list=["2020-01-01", "2020-01-03"],
        relevant_cols=["streamflow"],
    )
    assert isinstance(result, dict)
    assert "basin_1" in result and "basin_2" in result
    assert all(isinstance(df, pd.DataFrame) for df in result.values())


def test_start_hour_in_a_day_validation():
    """Test that start_hour_in_a_day validation works correctly for time interval format."""
    selfmadehydrodataset_path = SETTING["local_data_path"]["datasets-interim"]

    # Test: Non-standard time_unit like 6h should be rejected at init level
    # (SelfMadeHydroDataset only supports '1h', '3h', '1D', '8D')
    with pytest.raises(ValueError) as excinfo:
        dataset_6h = SelfMadeHydroDataset(
            data_path=selfmadehydrodataset_path,
            dataset_name="FDSources",
            time_unit=["6h"],
        )
    assert "time_unit must be one of" in str(excinfo.value)

    # Test that the cache_timeseries_xrdataset validation catches unsupported intervals
    # when they are passed directly to the function
    dataset = SelfMadeHydroDataset(
        data_path=selfmadehydrodataset_path, dataset_name="FDSources", time_unit=["3h"]
    )
    dataset.trange4cache = None
    dataset.offset_to_utc = False

    # Try to use an unsupported interval through kwargs
    # This should be caught by our new validation logic
    with pytest.raises(ValueError) as excinfo:
        dataset.cache_timeseries_xrdataset(
            time_units=["6h"], start_hour_in_a_day=2, batchsize=10
        )
    assert "only '3h' sub-daily interval is supported" in str(excinfo.value)

    # Test invalid start_hour_in_a_day value (should be 0-23)
    with pytest.raises(ValueError) as excinfo:
        dataset.cache_timeseries_xrdataset(
            time_units=["3h"], start_hour_in_a_day=25, batchsize=10
        )
    assert "must be an integer between 0 and 23" in str(excinfo.value)

    # Test invalid start_hour_in_a_day type
    with pytest.raises(ValueError) as excinfo:
        dataset.cache_timeseries_xrdataset(
            time_units=["3h"], start_hour_in_a_day="02:00:00", batchsize=10
        )
    assert "must be an integer between 0 and 23" in str(excinfo.value)


def test_start_hour_in_a_day_time_range():
    """Test that start_hour_in_a_day correctly sets the time range."""
    selfmadehydrodataset_path = SETTING["local_data_path"]["datasets-interim"]

    dataset = SelfMadeHydroDataset(
        data_path=selfmadehydrodataset_path, dataset_name="FDSources", time_unit=["3h"]
    )
    dataset.trange4cache = None
    dataset.offset_to_utc = False

    # Test with start_hour_in_a_day = 5
    # Expected: start with "05", end with "23" (05, 08, 11, 14, 17, 20, 23)
    try:
        dataset.cache_timeseries_xrdataset(
            time_units=["3h"], start_hour_in_a_day=5, batchsize=10
        )
    except Exception:
        # Ignore execution errors, we just want to check trange4cache
        pass

    # Check that trange4cache was set correctly
    assert dataset.trange4cache is not None
    assert "05" in dataset.trange4cache[0]
    assert "23" in dataset.trange4cache[1]


def test_start_hour_in_a_day_data_alignment(mocker):
    """Test that data alignment validation works correctly."""
    selfmadehydrodataset_path = SETTING["local_data_path"]["datasets-interim"]

    dataset = SelfMadeHydroDataset(
        data_path=selfmadehydrodataset_path, dataset_name="FDSources", time_unit=["3h"]
    )
    dataset.offset_to_utc = False

    # Mock data with hours starting at 00:00 (0, 3, 6, 9, 12, 15, 18, 21)
    mock_data = pd.DataFrame(
        {
            "time": pd.date_range("2020-01-01 00:00", periods=8, freq="3h"),
            "streamflow": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        }
    )

    mocker.patch("pandas.read_csv", return_value=mock_data)

    # Test 1: Correct alignment - should not raise error
    # Data starts at 00:00, so start_hour_in_a_day should be 0
    try:
        result = dataset.read_timeseries(
            object_ids=["test_basin"],
            t_range_list=["2020-01-01 00", "2020-01-02 00"],
            relevant_cols=["streamflow"],
            time_units=["3h"],
            start_hour_in_a_day=0,
        )
        # If we get here, the validation passed
        assert "3h" in result
    except ValueError as e:
        pytest.fail(f"Should not raise error with correct alignment: {e}")

    # Test 2: Incorrect alignment - should raise error
    # Data starts at 00:00, but start_hour_in_a_day is set to 2
    with pytest.raises(ValueError) as excinfo:
        dataset.read_timeseries(
            object_ids=["test_basin"],
            t_range_list=["2020-01-01 00", "2020-01-02 00"],
            relevant_cols=["streamflow"],
            time_units=["3h"],
            start_hour_in_a_day=2,
        )

    # Check error message content
    assert "Data time alignment error" in str(excinfo.value)
    assert "Please set start_hour_in_a_day to 0" in str(excinfo.value)
