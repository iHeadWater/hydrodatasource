import geopandas as gpd
import numpy as np
import pandas as pd
from geopandas import GeoDataFrame
from scipy.spatial import Voronoi
from shapely.geometry import Polygon
import hydrodatasource.configs.config as hdscc


def read_data(rainfall_data_paths: list, head='local', check_time=None):
    # Read rainfall CSV files
    rainfall_dfs = []
    check_time = pd.to_datetime(check_time, format='%Y-%m-%d %H:%M:%S')
    latest_date = pd.Timestamp.min  # initialize latest date as minimum Timestamp
    # Find latest date in CSV files
    for file in rainfall_data_paths:
        if head == 'local':
            df = pd.read_csv(file)
        elif head == 'minio':
            df = pd.read_csv(file, storage_options=hdscc.MINIO_PARAM)
        else:
            df = pd.DataFrame()
        first_row_date = pd.to_datetime(df.iloc[0]['TM'])
        if (first_row_date > latest_date) & (first_row_date <= check_time):
            latest_date = first_row_date
            rainfall_dfs.append(df)
    # Convert rainfall data and filter by latest date
    if len(rainfall_dfs) > 0:
        rainfall_df = pd.concat(rainfall_dfs).drop_duplicates().reset_index(drop=True)
        rainfall_df['TM'] = pd.to_datetime(rainfall_df['TM'])
        rainfall_df = rainfall_df[rainfall_df['TM'] >= latest_date]
    else:
        temp_range = pd.date_range('1990-01-01', '2038-12-31', freq='h')
        rainfall_df = pd.DataFrame({'TM': temp_range, 'DRP': np.repeat(0, len(temp_range.to_list()))})
    return rainfall_df


def calculate_voronoi_polygons(stations, basin_geom):
    """Calculate Voronoi polygons for each station."""
    bounding_box = basin_geom.envelope.exterior.coords
    points = np.array([point.coords[0] for point in stations.geometry])
    points_extended = np.concatenate((points, bounding_box))
    vor = Voronoi(points_extended)
    regions = [vor.regions[vor.point_region[i]] for i in range(len(points))]
    polygons = [Polygon(vor.vertices[region]).buffer(0) for region in regions if -1 not in region]
    polygons_gdf = gpd.GeoDataFrame(geometry=polygons, crs=stations.crs)
    polygons_gdf['station_id'] = stations['STCD'].astype(str).values
    polygons_gdf['original_area'] = polygons_gdf.geometry.area
    clipped_polygons_gdf = gpd.clip(polygons_gdf, basin_geom)
    clipped_polygons_gdf['clipped_area'] = clipped_polygons_gdf.geometry.area
    total_basin_area = basin_geom.area
    clipped_polygons_gdf['area_ratio'] = clipped_polygons_gdf['clipped_area'] / total_basin_area
    return clipped_polygons_gdf


def calculate_weighted_rainfall(voronoi_polygons, rainfall_data):
    voronoi_polygons['station_id'] = voronoi_polygons['station_id'].astype(str)
    if 'STCD' in rainfall_data.columns:
        rainfall_data['station_id'] = rainfall_data['STCD'].astype(str)
    else:
        rainfall_data['station_id'] = voronoi_polygons['station_id'].astype(str)
    merged_data = pd.merge(voronoi_polygons, rainfall_data, on='station_id')
    merged_data['weighted_rainfall'] = merged_data['DRP'] * merged_data['area_ratio']
    weighted_average_rainfall = merged_data.groupby('TM')['weighted_rainfall'].sum()
    return weighted_average_rainfall.reset_index()


def rainfall_average(basin: GeoDataFrame, stations_gdf: GeoDataFrame, pp_ids: list, check_time):
    # 如果新站对不上老时间，就会把老数据也砍掉
    check_time_path = f"{basin['BASIN_ID'][0]}_{check_time}_rainfall_mean.csv"
    s3_check_time_path = 's3://basins-origin/hour_data/1h/mean_data/' + check_time_path
    if hdscc.FS.exists(s3_check_time_path):
        return pd.read_csv(s3_check_time_path, storage_options=hdscc.MINIO_PARAM)
    else:
        rainfall_data_paths = [f's3://stations-origin/pp_stations/hour_data/1h/pp_{ppid}.csv' for ppid in pp_ids]
        rainfall_df = read_data(rainfall_data_paths, head='minio', check_time=check_time)
        stations_within_basin = gpd.sjoin(stations_gdf, basin)
        voronoi_polygons = calculate_voronoi_polygons(stations_within_basin, basin.geometry[0])
        average_rainfall = calculate_weighted_rainfall(voronoi_polygons, rainfall_df)
        # basin_name = os.path.splitext(os.path.basename(basin_shp_path))[0]
        # average_rainfall['basin_name'] = basin_name
        average_rainfall.to_csv(s3_check_time_path, index=False, storage_options=hdscc.MINIO_PARAM)
    return average_rainfall


'''
def plot_voronoi_polygons(original_polygons, clipped_polygons, basin):
    fig, (ax_original, ax_clipped) = plt.subplots(1, 2, figsize=(12, 6))
    original_polygons.plot(ax=ax_original, edgecolor='black')
    basin.boundary.plot(ax=ax_original, color='red')
    ax_original.set_title('Original Voronoi Polygons')
    clipped_polygons.plot(ax=ax_clipped, edgecolor='black')
    basin.boundary.plot(ax=ax_clipped, color='red')
    ax_clipped.set_title('Clipped Voronoi Polygons')
    plt.tight_layout()
    plt.show()
'''
