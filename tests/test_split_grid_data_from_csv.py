import os
import pathlib

import geopandas as gpd
import pandas as pd
import xarray as xr
from hydrotopo import ig_path
import hydrodatasource.configs.config as conf
from hydrodatasource.reader.spliter_grid import (
    query_path_from_metadata,
    generate_bbox_from_shp,
)


def test_generate_bbox_from_shp():
    basin_shp = "s3://basins-origin/basins_shp.zip"
    bbox = generate_bbox_from_shp(basin_shape_path=basin_shp, data_source="gpm")
    return bbox


def test_split_grid_data_from_single_basin_gpm():
    test_shp = "s3://basins-origin/basin_shapefiles/basin_USA_camels_12145500.zip"
    bbox = generate_bbox_from_shp(test_shp, data_source="gpm")
    time_start = "2018-06-05 01:00:00"
    time_end = "2018-06-05 02:00:00"
    tile_list = query_path_from_metadata(time_start, time_end, bbox, data_source="gpm")
    data_list = []
    for tile in tile_list:
        data_list.append(xr.open_dataset(conf.FS.open(tile)))
    print(data_list)
    return tile_list


def test_split_grid_data_from_single_basin_gfs():
    test_shp = "s3://basins-origin/basin_shapefiles/basin_USA_camels_01414500.zip"
    bbox = generate_bbox_from_shp(test_shp, data_source="gfs")
    time_start = "2022-01-03"
    time_end = "2022-01-03"
    tile_list = query_path_from_metadata(time_start, time_end, bbox, data_source="gfs")
    data_list = []
    for tile in tile_list:
        data_list.append(xr.open_dataset(conf.FS.open(tile)))
    print(data_list)
    return tile_list


def test_split_grid_data_from_single_basin_smap():
    test_shp = "s3://basins-origin/basin_shapefiles/basin_USA_camels_01414500.zip"
    bbox = generate_bbox_from_shp(test_shp, data_source="smap")
    time_start = "2016-02-02"
    time_end = "2016-02-02"
    tile_list = query_path_from_metadata(time_start, time_end, bbox, data_source="smap")
    data_list = []
    for tile in tile_list:
        data_list.append(xr.open_dataset(conf.FS.open(tile)))
    print(data_list)
    return tile_list


def test_split_grid_data_from_single_basin_era5():
    test_shp = "s3://basins-origin/basin_shapefiles/basin_CHN_songliao_10810201.zip"
    bbox = generate_bbox_from_shp(test_shp, data_source="era5_land")
    time_start = "2022-06-02"
    time_end = "2022-06-02"
    tile_list = query_path_from_metadata(
        time_start, time_end, bbox, data_source="era5_land"
    )
    data_list = []
    for tile in tile_list:
        data_list.append(xr.open_dataset(conf.FS.open(tile)))
    print(data_list)
    return tile_list


def test_read_topo_data():
    dams_shp = gpd.read_file(
        conf.FS.open("s3://reservoirs-origin/dams.zip"), engine="pyogrio"
    )
    network_shp = gpd.read_file(
        os.path.join(
            pathlib.Path(__file__).parent.parent,
            "data/river_network/songliao_cut_single.shp",
        ),
        engine="pyogrio",
    )
    index = dams_shp.index[dams_shp["ID"] == "zq_CHN_songliao_10310500"]
    paths = ig_path.find_edge_nodes(dams_shp, network_shp, index, "up")
    for station in paths:
        sta_id = dams_shp["ID"][dams_shp.index == station].to_list()[0]
        rr_path = "s3://reservoirs-origin/rr_stations/" + sta_id + ".csv"
        rr_df = pd.read_csv(rr_path, storage_options=conf.MINIO_PARAM)
        print(rr_df)
