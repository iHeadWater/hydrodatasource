"""
Author: liutiaxqabs 1498093445@qq.com
Date: 2024-12-12 11:04:10
LastEditTime: 2025-08-21 08:31:38
LastEditors: Wenyu Ouyang
Description: Test for basin_mean_rainfall.py
FilePath: \hydrodatasource\tests\test_basin_mean_rainfall.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import pytest
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
from hydrodatasource.processor.basin_mean_rainfall import (
    basin_mean_func,
    calculate_thiesen_polygons,
)


@pytest.fixture
def simple_basin():
    # Create a square basin polygon
    poly = Polygon([(0, 0), (0, 10), (10, 10), (10, 0)])
    return gpd.GeoDataFrame(
        {"BASIN_ID": ["test_basin"]}, geometry=[poly], crs="EPSG:4326"
    )


@pytest.fixture
def simple_stations():
    # Create 3 stations inside the basin
    points = [Point(2, 2), Point(8, 2), Point(5, 8)]
    stcds = ["A", "B", "C"]
    return gpd.GeoDataFrame({"STCD": stcds}, geometry=points, crs="EPSG:4326")


def test_calculate_thiesen_polygons_multiple_stations(simple_stations, simple_basin):
    clipped = calculate_thiesen_polygons(simple_stations, simple_basin)
    # Should return a GeoDataFrame with 3 polygons, one for each station
    assert isinstance(clipped, gpd.GeoDataFrame)
    assert len(clipped) == 3
    assert all(
        col in clipped.columns
        for col in ["STCD", "original_area", "clipped_area", "area_ratio"]
    )
    # Area ratios should sum to 1 (or very close)
    np.testing.assert_allclose(clipped["area_ratio"].sum(), 1.0, rtol=1e-6)


def test_calculate_thiesen_polygons_single_station(simple_basin):
    # Only one station
    gdf = gpd.GeoDataFrame({"STCD": ["A"]}, geometry=[Point(5, 5)], crs="EPSG:4326")
    clipped = calculate_thiesen_polygons(gdf, simple_basin)
    assert isinstance(clipped, gpd.GeoDataFrame)
    assert len(clipped) == 1
    assert np.isnan(clipped.iloc[0]["original_area"])
    assert np.isnan(clipped.iloc[0]["clipped_area"])
    assert clipped.iloc[0]["area_ratio"] == 1.0


def test_calculate_thiesen_polygons_no_station(simple_basin):
    # No stations
    gdf = gpd.GeoDataFrame({"STCD": []}, geometry=[], crs="EPSG:4326")
    clipped = calculate_thiesen_polygons(gdf, simple_basin)
    assert isinstance(clipped, gpd.GeoDataFrame)
    assert len(clipped) == 0


def test_calculate_thiesen_polygons_station_on_edge(simple_basin):
    # Station on the edge of the basin
    gdf = gpd.GeoDataFrame(
        {"STCD": ["A", "B"]}, geometry=[Point(0, 0), Point(10, 10)], crs="EPSG:4326"
    )
    clipped = calculate_thiesen_polygons(gdf, simple_basin)
    assert isinstance(clipped, gpd.GeoDataFrame)
    assert len(clipped) == 2
    np.testing.assert_allclose(clipped["area_ratio"].sum(), 1.0, rtol=1e-6)


def test_basin_mean_func_arithmetic_mean():
    # Simple arithmetic mean, no weights
    df = pd.DataFrame(
        {
            "st1": [1.0, 2.0, 3.0],
            "st2": [4.0, 5.0, 6.0],
        }
    )
    result = basin_mean_func(df)
    expected = pd.Series([2.5, 3.5, 4.5])
    pd.testing.assert_series_equal(
        result.reset_index(drop=True), expected, check_names=False
    )


def test_basin_mean_func_with_nan():
    # Arithmetic mean with NaN values
    df = pd.DataFrame(
        {
            "st1": [1.0, np.nan, 3.0],
            "st2": [4.0, 5.0, np.nan],
        }
    )
    result = basin_mean_func(df)
    expected = pd.Series([2.5, 5.0, 3.0])
    pd.testing.assert_series_equal(
        result.reset_index(drop=True), expected, check_names=False
    )


def test_basin_mean_func_all_nan_row():
    # Row with all NaN should return NaN
    df = pd.DataFrame(
        {
            "st1": [np.nan, 2.0],
            "st2": [np.nan, 4.0],
        }
    )
    result = basin_mean_func(df)
    assert np.isnan(result.iloc[0])
    assert result.iloc[1] == 3.0


def test_basin_mean_func_weighted_mean():
    # Weighted mean
    df = pd.DataFrame(
        {
            "st1": [1.0, 2.0, 3.0],
            "st2": [4.0, 5.0, 6.0],
        }
    )
    weights = {("st1", "st2"): (2 / 3, 1 / 3)}
    result = basin_mean_func(df, weights)
    # (1*2+4*1)/3 = 2.0, (2*2+5*1)/3 = 3.0, (3*2+6*1)/3 = 4.0
    expected = pd.Series([2.0, 3.0, 4.0])
    pd.testing.assert_series_equal(
        result.reset_index(drop=True), expected, check_names=False
    )


def test_basin_mean_func_multi_station_multi_weight():
    # 4 stations, 3 rows of data
    df = pd.DataFrame(
        {
            "st1": [1.0, 2.0, 3.0],
            "st2": [4.0, 5.0, 6.0],
            "st3": [7.0, 8.0, 9.0],
            "st4": [10.0, 11.0, 12.0],
        }
    )
    # First weight type: all stations
    weights1 = {("st1", "st2", "st3", "st4"): [0.1, 0.2, 0.3, 0.4]}
    result1 = basin_mean_func(df, weights1)
    expected1 = pd.Series(
        [
            1.0 * 0.1 + 4.0 * 0.2 + 7.0 * 0.3 + 10.0 * 0.4,
            2.0 * 0.1 + 5.0 * 0.2 + 8.0 * 0.3 + 11.0 * 0.4,
            3.0 * 0.1 + 6.0 * 0.2 + 9.0 * 0.3 + 12.0 * 0.4,
        ]
    )
    pd.testing.assert_series_equal(
        result1.reset_index(drop=True), expected1, check_names=False
    )

    # Second weight type: partial station combinations
    weights2 = {
        ("st1", "st2", "st3", "st4"): [0.25, 0.25, 0.25, 0.25],
        ("st1", "st2"): [0.6, 0.4],
        ("st3", "st4"): [0.7, 0.3],
    }
    # Create partial missing data scenario
    df2 = pd.DataFrame(
        {
            "st1": [1.0, np.nan, 3.0],
            "st2": [4.0, 5.0, np.nan],
            "st3": [7.0, 8.0, 9.0],
            "st4": [10.0, 11.0, 12.0],
        }
    )
    # First row has all data, second row only has st2, st3, st4, third row only has st1, st3, st4
    # But weights2 only defines (st1,st2) and (st3,st4), other combinations use equal weights
    result2 = basin_mean_func(df2, weights2)
    # First row: (1+4+7+10)/4 = 5.5
    # Second row: (5*0.6 + 8*0.7 + 11*0.3) / (0.6+0.7+0.3) = (3+5.6+3.3)/1.6 = 11.9/1.6=7.4375 (but actually key is (st2,st3,st4), undefined, use equal weights)
    # Actual second row: (5+8+11)/3=8.0
    # Third row: (3+9+12)/3=8.0
    expected2 = pd.Series(
        [
            (1.0 + 4.0 + 7.0 + 10.0) / 4,
            (5.0 + 8.0 + 11.0) / 3,
            (3.0 + 9.0 + 12.0) / 3,
        ]
    )
    pd.testing.assert_series_equal(
        result2.reset_index(drop=True), expected2, check_names=False
    )


def test_basin_mean_func_partial_weight_match():
    # 3 stations with partial weight combinations
    df = pd.DataFrame(
        {
            "A": [1.0, np.nan, 3.0],
            "B": [4.0, 5.0, np.nan],
            "C": [7.0, 8.0, 9.0],
        }
    )
    weights = {
        ("A", "B", "C"): [0.2, 0.3, 0.5],
        ("A", "C"): [0.4, 0.6],
        ("B", "C"): [0.7, 0.3],
    }
    result = basin_mean_func(df, weights)
    # First row: (1*0.2+4*0.3+7*0.5)=0.2+1.2+3.5=4.9
    # Second row: arithmetic mean of (5,8)=6.5
    # Third row: arithmetic mean of (3,9)=6.0
    expected = pd.Series([4.9, 6.5, 6.0])
    pd.testing.assert_series_equal(
        result.reset_index(drop=True), expected, check_names=False
    )
