"""
Author: Wenyu Ouyang
Date: 2024-03-23 15:10:23
LastEditTime: 2024-05-20 18:09:22
LastEditors: Wenyu Ouyang
Description: Test for utility functions
FilePath: \hydrodatasource\tests\test_utils.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import numpy as np
import xarray as xr
import pint
import pytest

from hydrodatasource.utils.utils import streamflow_unit_conv

ureg = pint.UnitRegistry()
ureg.force_ndarray_like = True  # or ureg.force_ndarray = True


# Test case for xarray input
@pytest.mark.parametrize(
    "streamflow, area, target_unit, inverse, expected",
    # [
    #     (
    #         xr.Dataset(
    #             {
    #                 "streamflow": xr.DataArray(
    #                     np.array([[100, 200], [300, 400]]), dims=["time", "basin"]
    #                 )
    #             }
    #         ),
    #         xr.Dataset({"area": xr.DataArray(np.array([1, 2]), dims=["basin"])}),
    #         "mm/d",
    #         False,
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
    #                 )
    #             }
    #         ),
    #     ),
    # ],
    # Add more test cases for xarray input
    [
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
    # [
    #     (
    #         np.array([100, 200]) * ureg.m**3 / ureg.s,
    #         np.array([1]) * ureg.km**2,
    #         "mm/d",
    #         False,
    #         np.array([8640.0, 17280.0]),
    #     ),
    # ],
    # [
    #     (
    #         np.array([100, 200]) * ureg.m**3 / ureg.s,
    #         np.array([1]) * ureg.km**2,
    #         "mm/3h",
    #         False,
    #         np.array([1080.0, 2160.0]),
    #     ),
    # ],
    # [
    #     (
    #         np.array([8640.0, 17280.0]) * ureg.mm / ureg.d,
    #         np.array([1]) * ureg.km**2,
    #         "m^3/s",
    #         True,
    #         np.array([100, 200]),
    #     ),
    # ],
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
    "streamflow, area, target_unit, inverse",
    [
        (None, np.array([2, 2, 2]), "mm/d", False),
        (np.array([10, 20, 30]), None, "mm/d", False),
        (np.array([10, 20, 30]), np.array([2, 2, 2]), "invalid_unit", False),
    ],
)
def test_streamflow_unit_conv_invalid_input(streamflow, area, target_unit, inverse):
    with pytest.raises(TypeError):
        streamflow_unit_conv(streamflow, area, target_unit, inverse)
