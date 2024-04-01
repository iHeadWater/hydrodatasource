from hydrodatasource.reader.spliter_grid import query_path_from_metadata, generate_bbox_from_shp
import xarray as xr
import hydrodatasource.configs.config as conf

def test_query_path_from_metadata_gpm():
    time_start = '2018-06-05 01:00:00'
    time_end = '2018-06-05 02:00:00'
    bbox = [-110, -69, 47, 26]
    paths = query_path_from_metadata(time_start, time_end, bbox, data_source='gpm')
    return paths


def test_query_path_from_metadata_gfs():
    # start_time (datetime, YY-mm-dd): The start date of the desired data.
    # end_time (datetime, YY-mm-dd): The end date of the desired data.
    time_start = '2022-01-03'
    time_end = '2022-01-03'
    bbox = [-110, -69, 47, 26]
    paths = query_path_from_metadata(time_start, time_end, bbox, data_source='gfs')
    return paths


def test_generate_bbox_from_shp():
    basin_shp = 's3://basins-origin/basins_shp.zip'
    mask, bbox = generate_bbox_from_shp(basin_shape_path=basin_shp)
    # shutil.rmtree('temp_mask')
    return mask, bbox


def test_split_grid_data_from_single_basin_gpm():
    test_shp = 's3://basins-origin/basin_shapefiles/basin_USA_camels_12145500.zip'
    mask, bbox = generate_bbox_from_shp(test_shp)
    time_start = '2018-06-05 01:00:00'
    time_end = '2018-06-05 02:00:00'
    tile_list = query_path_from_metadata(time_start, time_end, bbox, data_source='gpm')
    data_list = []
    for tile in tile_list:
        data_list.append(xr.open_dataset(conf.FS.open(tile)))
    print(data_list)
    return tile_list


def test_split_grid_data_from_single_basin_gfs():
    test_shp = 's3://basins-origin/basin_shapefiles/basin_USA_camels_01414500.zip'
    mask, bbox = generate_bbox_from_shp(test_shp)
    time_start = '2022-01-03'
    time_end = '2022-01-03'
    tile_list = query_path_from_metadata(time_start, time_end, bbox, data_source='gfs')
    data_list = []
    for tile in tile_list:
        data_list.append(xr.open_dataset(conf.FS.open(tile)))
    print(data_list)
    return tile_list
