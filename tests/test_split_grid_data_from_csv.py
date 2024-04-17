import os
import pathlib

import geopandas as gpd
import pandas as pd
import xarray as xr
from dijkstra_conda import ig_path

import hydrodatasource.configs.config as conf
from hydrodatasource.reader.spliter_grid import (
    query_path_from_metadata,
    generate_bbox_from_shp,
    merge_with_spatial_average,
)


def test_query_path_from_metadata_gpm():
    time_start = "2018-06-05 01:00:00"
    time_end = "2018-06-05 02:00:00"
    bbox = [-110, -69, 47, 26]
    paths = query_path_from_metadata(time_start, time_end, bbox, data_source="gpm")
    return paths


def test_query_path_from_metadata_gfs():
    # start_time (datetime, YY-mm-dd): The start date of the desired data.
    # end_time (datetime, YY-mm-dd): The end date of the desired data.
    time_start = "2022-01-03"
    time_end = "2022-01-03"
    bbox = [-110, -69, 47, 26]
    paths = query_path_from_metadata(time_start, time_end, bbox, data_source="gfs")
    return paths


def test_query_path_from_metadata_smap():
    # start_time (datetime, YY-mm-dd): The start date of the desired data.
    # end_time (datetime, YY-mm-dd): The end date of the desired data.
    time_start = "2016-02-02"
    time_end = "2016-02-02"
    bbox = [-110, -69, 47, 26]
    paths = query_path_from_metadata(time_start, time_end, bbox, data_source="smap")
    return paths


def test_generate_bbox_from_shp():
    basin_shp = "s3://basins-origin/basins_shp.zip"
    mask, bbox = generate_bbox_from_shp(basin_shape_path=basin_shp, dataname='gfs')
    # shutil.rmtree('temp_mask')
    return mask, bbox


def test_split_grid_data_from_single_basin_gpm():
    test_shp = "s3://basins-origin/basin_shapefiles/basin_USA_camels_12145500.zip"
    mask, bbox = generate_bbox_from_shp(test_shp, 'gpm')
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
    mask, bbox = generate_bbox_from_shp(test_shp, 'gfs')
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
    # smap分辨率与gfs/era5/gpm不同，故mask是错的，姑且用era5代替
    mask, bbox = generate_bbox_from_shp(test_shp, 'era5')
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
    mask, bbox = generate_bbox_from_shp(test_shp, 'era5')
    time_start = "2022-06-02"
    time_end = "2022-06-02"
    tile_list = query_path_from_metadata(time_start, time_end, bbox, data_source="era5_land")
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


def merge_with_spatial_average():
    # 暂无数据 求某个流域gpm/gfs/smap面平均,并合并为mean_forcing.nc
    gpm_path = (
        "basin-origin/hour_data/1h/grid_data/grid_gpm_data/grid_gpm_CHN_21401550.nc"
    )

    gfs_path = (
        "basin-origin/hour_data/1h/grid_data/grid_gfs_data/grid_gfs_CHN_21401550.nc"
    )

    smap_path = (
        "basin-origin/hour_data/1h/grid_data/grid_smap_data/grid_smap_CHN_21401550.nc"
    )

    out_path = "basin-origin/hour_data/1h/mean_data/mean_data_forcing/mean_forcing_CHN_21401550.nc"
    merged_ds = merge_with_spatial_average(gpm_path, gfs_path, smap_path, out_path)
    print(merged_ds)
