import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
from scipy.spatial import Voronoi
import numpy as np
import matplotlib.pyplot as plt
from geopandas.tools import sjoin
import os



def read_data(stations_csv_path: str,
              basin_shp_path: str,
              rainfall_data_folder: str) -> Tuple[pd.DataFrame, gpd.GeoDataFrame, pd.DataFrame, Set[str]]:
    """Reads station and rainfall data.

    Args:
        stations_csv_path: Path to CSV file containing station data.
        basin_shp_path: Path to shapefile containing basin data.
        rainfall_data_folder: Path to folder containing rainfall CSV files.

    Returns:
        A tuple containing station data, basin data, rainfall data, and a set of station IDs.
    """
    # Read station data
    stations_df = pd.read_csv(stations_csv_path)
    stations_df.dropna(subset=['LON', 'LAT'], inplace=True)

    # Read basin shapefile
    basin = gpd.read_file(basin_shp_path)

    # Read rainfall CSV files
    rainfall_files = [f for f in os.listdir(rainfall_data_folder) if f.endswith('.csv')]
    rainfall_dfs, station_ids = [], set()
    latest_date = pd.Timestamp.min  # initialize latest date as minimum Timestamp

    # Find latest date in CSV files
    for file in rainfall_files:
        df = pd.read_csv(os.path.join(rainfall_data_folder, file), nrows=1)
        first_row_date = pd.to_datetime(df.iloc[0]['TM'])
        if first_row_date > latest_date:
            latest_date = first_row_date

    # Read and append rainfall data
    for file in rainfall_files:
        df = pd.read_csv(os.path.join(rainfall_data_folder, file))
        rainfall_dfs.append(df)
        station_ids.update(df['STCD'].astype(str))

    # Convert rainfall data and filter by latest date
    rainfall_df = pd.concat(rainfall_dfs).drop_duplicates().reset_index(drop=True)
    rainfall_df['TM'] = pd.to_datetime(rainfall_df['TM'])
    rainfall_df = rainfall_df[rainfall_df['TM'] >= latest_date]

    return stations_df, basin, rainfall_df, station_ids

def process_stations(stations_df, basin, station_ids):
    stations_df = stations_df[stations_df['STCD'].astype(str).isin(station_ids)]
    stations_gdf = gpd.GeoDataFrame(
        stations_df, 
        geometry=[Point(xy) for xy in zip(stations_df.LON, stations_df.LAT)],
        crs="EPSG:4326"
    )
    stations_gdf = stations_gdf.to_crs(basin.crs)
    return gpd.sjoin(stations_gdf, basin, how="inner")

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



# 主函数
def main(station_csv_path, basin_shp_path, rainfall_data_dir):
    stations_df, basin, rainfall_df, station_ids = read_data(station_csv_path, basin_shp_path, rainfall_data_dir)
    stations_within_basin = process_stations(stations_df, basin, station_ids)
    voronoi_polygons = calculate_voronoi_polygons(stations_within_basin, basin)
    average_rainfall = calculate_weighted_rainfall(voronoi_polygons, rainfall_df)
    basin_name = os.path.splitext(os.path.basename(basin_shp_path))[0]
    average_rainfall['basin_name'] = basin_name
    average_rainfall.to_csv(f"{basin_name}_mean_rainfall.csv", index=False)

# 使用示例
stations_csv_path = "/home/liutianxv1/计算流域面平均/pp_stations.csv"
basin_shp_path = "/home/liutianxv1/0318检测站点基础信息/流域合并/allbasins.shp"#"/home/liutianxv1/碧流河秋季数据分析/泰森多边形/碧流河流域.shp"#
rainfall_data_folder = "/home/liutianxv1/计算流域面平均/data/"
main(stations_csv_path, basin_shp_path, rainfall_data_folder)