import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import Voronoi
from shapely.geometry import Point, Polygon
import hydrodatasource.configs.config as hdscc


def read_data(rainfall_data_paths: list, head='local'):
    # Read rainfall CSV files
    rainfall_dfs = []
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
        if first_row_date > latest_date:
            latest_date = first_row_date
        rainfall_dfs.append(df)
    # Convert rainfall data and filter by latest date
    rainfall_df = pd.concat(rainfall_dfs).drop_duplicates().reset_index(drop=True)
    rainfall_df['TM'] = pd.to_datetime(rainfall_df['TM'])
    rainfall_df = rainfall_df[rainfall_df['TM'] >= latest_date]
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
    rainfall_data['station_id'] = rainfall_data['STCD'].astype(str)
    merged_data = pd.merge(voronoi_polygons, rainfall_data, on='station_id')
    merged_data['weighted_rainfall'] = merged_data['DRP'] * merged_data['area_ratio']
    weighted_average_rainfall = merged_data.groupby('TM')['weighted_rainfall'].sum()
    return weighted_average_rainfall.reset_index()

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

