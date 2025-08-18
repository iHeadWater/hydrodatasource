"""
Author: Wenyu Ouyang
Date: 2024-03-23 15:10:23
LastEditTime: 2025-08-18 15:47:38
LastEditors: Wenyu Ouyang
Description: Test for utility functions
FilePath: \hydrodatasource\tests\test_utils.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import numpy as np
import xarray as xr
import pint
import pytest

from hydrodatasource.utils.utils import (
    streamflow_unit_conv,
    minio_file_list,
    is_minio_folder,
)

ureg = pint.UnitRegistry()
ureg.force_ndarray_like = True  # or ureg.force_ndarray = True


# Test case for xarray input
@pytest.mark.parametrize(
    "streamflow, area, target_unit, inverse, expected",
    [
        # Test case 1: Convert to mm/d (daily)
        (
            xr.Dataset(
                {
                    "streamflow": xr.DataArray(
                        np.array([[100, 200], [300, 400]]), dims=["time", "basin"]
                    )
                }
            ),
            xr.Dataset({"area": xr.DataArray(np.array([1, 2]), dims=["basin"])}),
            "mm/d",
            False,
            xr.Dataset(
                {
                    "streamflow": xr.DataArray(
                        np.array(
                            [
                                [8640.0, 8640.0],
                                [25920.0, 17280.0],
                            ]
                        ),
                        dims=["time", "basin"],
                    )
                }
            ),
        ),
        # Test case 2: Convert to mm/3h (3-hourly)
        (
            xr.Dataset(
                {
                    "streamflow": xr.DataArray(
                        np.array([[100, 200], [300, 400]]), dims=["time", "basin"]
                    )
                }
            ),
            xr.Dataset({"area": xr.DataArray(np.array([1, 2]), dims=["basin"])}),
            "mm/3h",
            False,
            xr.Dataset(
                {
                    "streamflow": xr.DataArray(
                        np.array(
                            [
                                [1080.0, 1080.0],
                                [3240.0, 2160.0],
                            ]
                        ),
                        dims=["time", "basin"],
                    )
                }
            ),
        ),
    ],
)
def test_streamflow_unit_conv_xarray(streamflow, area, target_unit, inverse, expected):
    # Attaching units using pint
    streamflow["streamflow"] = streamflow["streamflow"] * ureg.m**3 / ureg.s
    area["area"] = area["area"] * ureg.km**2

    result = streamflow_unit_conv(streamflow, area, target_unit, inverse)
    xr.testing.assert_allclose(result, expected)


@pytest.mark.parametrize(
    "streamflow, area, target_unit, inverse, expected",
    # [
    #     (
    #         xr.Dataset(
    #             {
    #                 "streamflow": xr.DataArray(
    #                     np.array(
    #                         [
    #                             [8640.0, 8640.0],
    #                             [25920.0, 17280.0],
    #                         ]
    #                     ),
    #                     dims=["time", "basin"],
    #                     attrs={"units": "mm/d"},
    #                 )
    #             }
    #         ),
    #         xr.Dataset(
    #             {
    #                 "area": xr.DataArray(
    #                     np.array([1, 2]), dims=["basin"], attrs={"units": "km^2"}
    #                 )
    #             }
    #         ),
    #         "m^3/s",
    #         True,
    #         xr.Dataset(
    #             {
    #                 "streamflow": xr.DataArray(
    #                     np.array([[100, 200], [300, 400]]),
    #                     dims=["time", "basin"],
    #                 )
    #             }
    #         ),
    #     ),
    # ],
    [
        (
            xr.Dataset(
                {
                    "streamflow": xr.DataArray(
                        np.array(
                            [
                                [1080.0, 1080.0],
                                [3240.0, 2160.0],
                            ]
                        ),
                        dims=["time", "basin"],
                        attrs={"units": "mm/3h"},
                    )
                }
            ),
            xr.Dataset(
                {
                    "area": xr.DataArray(
                        np.array([1, 2]), dims=["basin"], attrs={"units": "km^2"}
                    )
                }
            ),
            "m^3/s",
            True,
            xr.Dataset(
                {
                    "streamflow": xr.DataArray(
                        np.array([[100, 200], [300, 400]]),
                        dims=["time", "basin"],
                    )
                }
            ),
        ),
    ],
)
def test_streamflow_unit_conv_xarray_inverse(
    streamflow, area, target_unit, inverse, expected
):
    result = streamflow_unit_conv(streamflow, area, target_unit, inverse)
    xr.testing.assert_allclose(result, expected)


# Test case for numpy and pandas input
@pytest.mark.parametrize(
    "streamflow, area, target_unit, inverse, expected",
    [
        (
            np.array([1080.0, 2160.0]) * ureg.mm / ureg.h / 3,
            np.array([1]) * ureg.km**2,
            "m^3/s",
            True,
            np.array([100, 200]),
        ),
    ],
)
def test_streamflow_unit_conv_np_pd(streamflow, area, target_unit, inverse, expected):
    result = streamflow_unit_conv(streamflow, area, target_unit, inverse)
    np.testing.assert_array_almost_equal(result, expected)


# Test case for invalid input type
@pytest.mark.parametrize(
    "streamflow, area, target_unit, inverse, expected_error",
    [
        # streamflow is None should raise ValueError
        (None, np.array([2, 2, 2]), "mm/d", False, ValueError),
        # numpy streamflow without source_unit should raise ValueError
        (np.array([10, 20, 30]), np.array([2, 2, 2]), "mm/d", False, ValueError),
        # invalid target_unit with valid inputs should raise ValueError
        (
            np.array([10, 20, 30]),
            np.array([2, 2, 2]),
            "invalid_unit",
            False,
            ValueError,
        ),
    ],
)
def test_streamflow_unit_conv_invalid_input(
    streamflow, area, target_unit, inverse, expected_error
):
    with pytest.raises(expected_error):
        streamflow_unit_conv(streamflow, area, target_unit, inverse)


# Test case for numpy arrays with source_unit and area_unit parameters
@pytest.mark.parametrize(
    "streamflow, area, source_unit, target_unit, area_unit, inverse, expected",
    [
        # Test case 1: m³/s to mm/3h conversion (numpy array, area without units)
        (
            np.array([100.0, 200.0, 300.0]),  # m³/s data
            np.array([1000]),  # km² (no units)
            "m^3/s",
            "mm/3h",
            "km^2",
            False,
            np.array([1.08, 2.16, 3.24]),  # 100*3600*3/(1000*1e6)*1000 = 1.08
        ),
        # Test case 2: mm/3h to m³/s inverse conversion (numpy array, area without units)
        (
            np.array([1.08, 2.16, 3.24]),  # mm/3h data
            np.array([1000]),  # km² (no units)
            "mm/3h",
            "m^3/s",
            "km^2",
            True,
            np.array([100.0, 200.0, 300.0]),  # 1.08*1000*1e6/(3600*3)/1000 = 100
        ),
        # Test case 3: m³/s to mm/6h conversion with different area unit
        (
            np.array([50.0, 100.0, 150.0]),  # m³/s data
            np.array([500000]),  # m² (no units)
            "m^3/s",
            "mm/6h",
            "m^2",
            False,
            np.array([2160.0, 4320.0, 6480.0]),  # 50*3600*6/500000*1000 = 2160
        ),
    ],
)
def test_streamflow_unit_conv_numpy_no_area_units(
    streamflow, area, source_unit, target_unit, area_unit, inverse, expected
):
    """Test conversion using source_unit and area_unit parameters for numpy arrays"""
    result = streamflow_unit_conv(
        streamflow, area, target_unit, inverse, source_unit, area_unit
    )
    # Check that result is a numpy array
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_almost_equal(result, expected, decimal=6)


# Test case for identical units (early return)
@pytest.mark.parametrize(
    "streamflow, area, target_unit, source_unit, inverse",
    [
        # Test case 1: xarray with identical units in attrs
        (
            xr.Dataset(
                {
                    "streamflow": xr.DataArray(
                        np.array([[100, 200], [300, 400]]),
                        dims=["time", "basin"],
                        attrs={"units": "mm/d"},
                    )
                }
            ),
            xr.Dataset(
                {
                    "area": xr.DataArray(
                        np.array([1, 2]), dims=["basin"], attrs={"units": "km^2"}
                    )
                }
            ),
            "mm/d",
            None,
            False,
        ),
        # Test case 2: xarray with identical units in attrs (custom unit)
        (
            xr.Dataset(
                {
                    "streamflow": xr.DataArray(
                        np.array([[100, 200], [300, 400]]),
                        dims=["time", "basin"],
                        attrs={"units": "mm/3h"},
                    )
                }
            ),
            xr.Dataset(
                {
                    "area": xr.DataArray(
                        np.array([1, 2]), dims=["basin"], attrs={"units": "km^2"}
                    )
                }
            ),
            "mm/3h",
            None,
            True,  # Even with wrong inverse, should return early
        ),
        # Test case 3: numpy array with explicit source_unit same as target_unit
        (
            np.array([100.0, 200.0, 300.0]),
            np.array([1000]),
            "m^3/s",
            "m^3/s",
            False,
        ),
        # Test case 4: pint quantity with identical units
        # Note: For pint quantities, the function extracts units automatically
        # and when units are identical, it should return early with magnitude only
        (
            np.array([100.0, 200.0, 300.0]) * ureg.m**3 / ureg.s,
            np.array([1000]) * ureg.km**2,
            "m^3/s",
            None,
            False,
        ),
        # Test case 5: normalized units (m3/s vs m^3/s)
        (
            np.array([100.0, 200.0, 300.0]),
            np.array([1000]),
            "m^3/s",
            "m3/s",  # Different notation but same unit
            True,
        ),
    ],
)
def test_streamflow_unit_conv_identical_units(
    streamflow, area, target_unit, source_unit, inverse
):
    """Test that function returns original data when source and target units are identical"""
    result = streamflow_unit_conv(streamflow, area, target_unit, inverse, source_unit)

    # For xarray, should return exact same object (early return)
    if isinstance(streamflow, xr.Dataset):
        assert result is streamflow, "Should return original xarray dataset object"
    # For numpy arrays, should return same values
    elif isinstance(streamflow, np.ndarray):
        np.testing.assert_array_equal(result, streamflow)
    # For pint quantities, early return should return original streamflow object
    elif isinstance(streamflow, pint.Quantity):
        assert result is streamflow, "Should return original pint quantity object"


# Test case for inverse parameter validation
@pytest.mark.parametrize(
    "streamflow, area, target_unit, source_unit, inverse, should_raise",
    [
        # Valid conversions - should not raise
        (
            np.array([100.0, 200.0, 300.0]),
            np.array([1000]),
            "mm/d",
            "m^3/s",
            False,  # volume -> depth: inverse=False
            False,
        ),
        (
            np.array([100.0, 200.0, 300.0]),
            np.array([1000]),
            "m^3/s",
            "mm/d",
            True,  # depth -> volume: inverse=True
            False,
        ),
        # Invalid conversions - should raise ValueError
        (
            np.array([100.0, 200.0, 300.0]),
            np.array([1000]),
            "mm/d",
            "m^3/s",
            True,  # volume -> depth: should be inverse=False
            True,
        ),
        (
            np.array([100.0, 200.0, 300.0]),
            np.array([1000]),
            "m^3/s",
            "mm/d",
            False,  # depth -> volume: should be inverse=True
            True,
        ),
    ],
)
def test_streamflow_unit_conv_inverse_validation(
    streamflow, area, target_unit, source_unit, inverse, should_raise
):
    """Test validation of inverse parameter consistency with unit conversion direction"""
    if should_raise:
        with pytest.raises(ValueError, match="Inverse parameter.*inconsistent"):
            streamflow_unit_conv(streamflow, area, target_unit, inverse, source_unit)
    else:
        # Should not raise an error
        result = streamflow_unit_conv(
            streamflow, area, target_unit, inverse, source_unit
        )
        assert isinstance(result, np.ndarray)


@pytest.mark.skip(reason="MinIO tests are optional and require server connection")
def test_minio_file_list():
    minio_folder_url = "s3://basins-interim/timeseries/1D"
    file_list = minio_file_list(minio_folder_url)
    print(file_list)


@pytest.mark.skip(reason="MinIO tests are optional and require server connection")
def test_is_minio_folder():
    minio_folder_url = "s3://basins-interim/timeseries/1D"
    print(is_minio_folder(minio_folder_url))
    minio_folder_url = "s3://basins-interim/timeseries/1D_units_info.json"
    print(is_minio_folder(minio_folder_url))
