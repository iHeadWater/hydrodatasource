import os

import geopandas as gpd

import hydrodatasource.configs.config as hdscc
from hydrodatasource.processor.basin_mean_rainfall import calculate_voronoi_polygons, calculate_weighted_rainfall, \
    read_data


def test_rainfall_average():
    basin_shp_path = 's3://basins-origin/basin_shapefiles/basin_CHN_songliao_21401550.zip'
    rainfall_stations = 's3://stations-origin/stations_list/pp_stations.zip'
    pp_ids = ['pp_CHN_biliuhezijian_4002.csv', 'pp_CHN_biliuhezijian_4003.csv', 'pp_CHN_biliuhezijian_4006.csv',
              'pp_CHN_biliuhezijian_4007.csv', 'pp_CHN_biliuhezijian_4008.csv']
    rainfall_data_paths = ['s3://stations-origin/pp_stations/hour_data/1h/' + ppid for ppid in pp_ids]
    rainfall_df = read_data(rainfall_data_paths, head='minio')
    basin = gpd.read_file(hdscc.FS.open(basin_shp_path))
    stations_gdf = gpd.read_file(hdscc.FS.open(rainfall_stations))
    stations_within_basin = gpd.sjoin(stations_gdf, basin)
    voronoi_polygons = calculate_voronoi_polygons(stations_within_basin, basin.geometry[0])
    average_rainfall = calculate_weighted_rainfall(voronoi_polygons, rainfall_df)
    basin_name = os.path.splitext(os.path.basename(basin_shp_path))[0]
    average_rainfall['basin_name'] = basin_name
    return average_rainfall
