"""
Author: Wenyu Ouyang
Date: 2023-10-25 15:16:21
LastEditTime: 2025-01-07 21:14:20
LastEditors: Wenyu Ouyang
Description: Tests for reading public reservoir datasets
FilePath: \hydrodatasource\tests\test_reservoirs.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import pytest
import geopandas as gpd
from hydrodatasource.reader.reservoir_datasets import Crd


@pytest.fixture
def crd_instance(tmp_path):
    # Create a temporary directory and files for testing
    data_path = tmp_path / "data"
    data_path.mkdir()
    all_rsvrs_dir = data_path / "CRD_v11_all_reservoirs"
    all_rsvrs_dir.mkdir()
    all_rsvrs_shpfile = all_rsvrs_dir / "CRD_v11_all_reservoirs.shp"

    # Create an empty GeoDataFrame and save it as a shapefile
    gdf = gpd.GeoDataFrame({"geometry": []})
    gdf.to_file(all_rsvrs_shpfile)

    return Crd(data_path)


def test_read_reservoir_info(crd_instance):
    result = crd_instance.read_reservoir_info()
    assert isinstance(result, gpd.GeoDataFrame)
    assert result.empty
