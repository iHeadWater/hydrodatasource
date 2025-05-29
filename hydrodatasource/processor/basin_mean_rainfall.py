import geopandas as gpd
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.spatial import Voronoi
from shapely.geometry import Polygon
import hydrodatasource.configs.config as hdscc


def read_data(rainfall_data_paths: list, head="local", check_time=None):
    # Read rainfall CSV files
    rainfall_dfs = []
    check_time = pd.to_datetime(check_time, format="%Y-%m-%d %H:%M:%S")
    latest_date = pd.Timestamp.min  # initialize latest date as minimum Timestamp
    # Find latest date in CSV files
    for file in rainfall_data_paths:
        if head == "local":
            df = pd.read_csv(file)
        elif head == "minio":
            df = pd.read_csv(file, storage_options=hdscc.MINIO_PARAM)
        else:
            df = pd.DataFrame()
        first_row_date = pd.to_datetime(df.iloc[0]["TM"])
        if (first_row_date > latest_date) & (first_row_date <= check_time):
            latest_date = first_row_date
            rainfall_dfs.append(df)
    # Convert rainfall data and filter by latest date
    if rainfall_dfs:
        rainfall_df = pd.concat(rainfall_dfs).drop_duplicates().reset_index(drop=True)
        rainfall_df["TM"] = pd.to_datetime(rainfall_df["TM"])
        rainfall_df = rainfall_df[rainfall_df["TM"] >= latest_date]
    else:
        temp_range = pd.date_range("1990-01-01", "2038-12-31", freq="h")
        rainfall_df = pd.DataFrame(
            {"TM": temp_range, "DRP": np.repeat(0, len(temp_range.to_list()))}
        )
    return rainfall_df


def calculate_thiesen_polygons(stations, basin):
    """
    Calculate Thiessen polygons and clip to basin boundary.

    Parameters:
    ------------
    stations: GeoDataFrame
        stations within the basin
    basin: GeoDataFrame
        basin shapefile

    Returns:
    ---------
    clipped_polygons: GeoDataFrame
        a GeoDataFrame containing the clipped Voronoi polygons with area_ratio as a column
    """
    if len(stations) < 2:
        stations["original_area"] = np.nan
        stations["clipped_area"] = np.nan
        stations["area_ratio"] = 1.0
        return stations

    # get the minimum and maximum coordinates of the basin boundary, and build the bounding box
    x_min, y_min, x_max, y_max = basin.total_bounds

    # extend the bounding box
    x_min -= 1.0 * (x_max - x_min)
    x_max += 1.0 * (x_max - x_min)
    y_min -= 1.0 * (y_max - y_min)
    y_max += 1.0 * (y_max - y_min)

    bounding_box = np.array(
        [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
    )

    # extract the coordinates of the stations
    points = np.array([point.coords[0] for point in stations.geometry])

    # combine the coordinates of the stations with the bounding box points, ensuring that the Voronoi polygons cover the entire basin
    points_extended = np.concatenate((points, bounding_box), axis=0)

    # calculate the Voronoi diagram
    vor = Voronoi(points_extended)

    # extract the Voronoi region corresponding to each point
    regions = [vor.regions[vor.point_region[i]] for i in range(len(points))]

    # generate polygons
    polygons = [
        Polygon([vor.vertices[i] for i in region if i != -1])
        for region in regions
        if -1 not in region
    ]

    # create a GeoDataFrame
    gdf_polygons = gpd.GeoDataFrame(geometry=polygons, crs=stations.crs)
    gdf_polygons["STCD"] = stations["STCD"].values
    gdf_polygons["original_area"] = gdf_polygons.geometry.area

    # clip the polygons to the basin boundary
    clipped_polygons = gpd.clip(gdf_polygons, basin)
    clipped_polygons["clipped_area"] = clipped_polygons.geometry.area
    clipped_polygons["area_ratio"] = (
        clipped_polygons["clipped_area"] / clipped_polygons["clipped_area"].sum()
    )

    return clipped_polygons


def calculate_voronoi_polygons(stations, basin_geom):
    """
    @deprecated

    Previous version of calculate_thiesen_polygons.
    Deprecated in favor of calculate_thiesen_polygons.
    Deprecated since version 0.0.11: Use calculate_thiesen_polygons instead.

    Parameters
    ----------
    stations : GeoDataFrame
        stations within the basin
    basin_geom : GeoDataFrame
        basin shapefile

    Returns
    -------
    clipped_polygons_gdf : GeoDataFrame
        clipped voronoi polygons
    """

    bounding_box = basin_geom.envelope.exterior.coords
    points = np.array([point.coords[0] for point in stations.geometry])
    points_extended = np.concatenate((points, bounding_box))
    vor = Voronoi(points_extended)
    regions = [vor.regions[vor.point_region[i]] for i in range(len(points))]
    polygons = [
        Polygon(vor.vertices[region]).buffer(0)
        for region in regions
        if -1 not in region
    ]
    polygons_gdf = gpd.GeoDataFrame(geometry=polygons, crs=stations.crs)
    polygons_gdf["station_id"] = stations["STCD"].astype(str).values
    polygons_gdf["original_area"] = polygons_gdf.geometry.area
    clipped_polygons_gdf = gpd.clip(polygons_gdf, basin_geom)
    clipped_polygons_gdf["clipped_area"] = clipped_polygons_gdf.geometry.area
    total_basin_area = basin_geom.area
    clipped_polygons_gdf["area_ratio"] = (
        clipped_polygons_gdf["clipped_area"] / total_basin_area
    )
    return clipped_polygons_gdf


def calculate_weighted_rainfall(
    station_weights,
    rainfall_df,
    station_id_name="STCD",
    rainfall_name="DRP",
    time_name="TM",
):
    """
    Calculate weighted average rainfall.

    @deprecated
    Deprecated in favor of basin_mean_func.
    Deprecated since version 0.0.11: Use basin_mean_func instead.

    Parameters:
    ------------
    station_weights
        the weight of each station.
    rainfall_df
        rainfall data DataFrame.

    Returns:
    ---------
    weighted_average_rainfall
        weighted average rainfall DataFrame.
    """
    station_weights[station_id_name] = station_weights[station_id_name].astype(str)

    # merge thiesen polygons and rainfall data
    merged_data = pd.merge(station_weights, rainfall_df, on=station_id_name)

    # calculate weighted rainfall
    merged_data["weighted_rainfall"] = (
        merged_data[rainfall_name] * merged_data["area_ratio"]
    )

    return merged_data.groupby(time_name)["weighted_rainfall"].sum().reset_index()


def basin_mean_func(df, weights_dict=None):
    """
    Generic basin averaging method that supports both arithmetic mean and weighted mean (e.g. Thiessen polygon weights)

    Parameters
    ----------
    df : DataFrame
        Time series DataFrame for multiple stations, with station names as column names;
        each column should be a time series of rainfall data for a specific station
    weights_dict : dict, optional
        Dictionary with tuple of station names as keys and list of weights as values.
        If None, arithmetic mean is used.

    NOTE: the keys of list must be in the same order as the columns of df.
        hence, an easy way is you give your df with a sorted column names and then
        use the same order to create the keys of weights_dict.
        for example:
        weights_dict = {
            ("st1", "st2"): [0.6, 0.4],
            ("st3", "st4"): [0.5, 0.5],
        }
        df = df[["st1", "st2", "st3", "st4"]]
        then the keys of weights_dict must be in the same order as the columns of df.

    Returns
    -------
    Series
        Basin-averaged time series
    """
    if not weights_dict:
        return df.mean(axis=1, skipna=True)
    # check if the keys of weights_dict are in the same order as the columns of df
    for key in weights_dict.keys():
        # Get indices of elements in key that exist in df.columns
        col_indices = [list(df.columns).index(col) for col in key if col in df.columns]
        # Check if indices are in ascending order, i.e. if the order matches
        if col_indices != sorted(col_indices):
            raise AssertionError(
                "The station order in each weights_dict key must match the order in df.columns"
            )

    def weighted_mean(row):
        valid_cols = [
            col for col, val in zip(df.columns, row.values) if not np.isnan(val)
        ]
        key = tuple(sorted(valid_cols))
        if not valid_cols:
            return np.nan
        weights = weights_dict.get(key)
        if weights is None:
            # If no weights are provided for this combination of stations, just use equal weights
            weights = np.ones(len(valid_cols)) / len(valid_cols)
        vals = [row[col] for col in valid_cols]
        return np.sum(np.array(vals) * np.array(weights))

    return df.apply(weighted_mean, axis=1)


def plot_voronoi_polygons(original_polygons, clipped_polygons, basin):
    fig, (ax_original, ax_clipped) = plt.subplots(1, 2, figsize=(12, 6))
    _plot_voronoi_polygons(
        original_polygons, ax_original, basin, "Original Voronoi Polygons"
    )
    _plot_voronoi_polygons(
        clipped_polygons, ax_clipped, basin, "Clipped Voronoi Polygons"
    )
    plt.tight_layout()
    plt.show()


def _plot_voronoi_polygons(arg0, ax, basin, arg3):
    arg0.plot(ax=ax, edgecolor="black")
    basin.boundary.plot(ax=ax, color="red")
    ax.set_title(arg3)


def stations_within_basin(basin_gdf, station_gdf, buffer_m=0, basin_crs_epsg=3857):
    """
    Get stations within the buffered basin boundary
    Parameters
    ----------
    basin_gdf : GeoDataFrame
        GeoDataFrame containing the basin shapefile
    station_gdf : GeoDataFrame
        GeoDataFrame containing the station shapefile
    buffer_m : float
        Buffer distance in meters, default 0
    basin_crs_epsg : int
        EPSG code for projected coordinate system, default 3857 (in meters)
    Returns
    -------
    GeoDataFrame
        Stations within the buffered basin boundary
    """
    # Project to coordinate system in meters
    basin_proj = basin_gdf.to_crs(epsg=basin_crs_epsg)
    station_proj = station_gdf.to_crs(epsg=basin_crs_epsg)
    # Add buffer to basin
    basin_proj = basin_proj.copy()
    basin_proj["geometry"] = basin_proj.geometry.buffer(buffer_m)
    # Convert back to original coordinate system
    basin_buffered = basin_proj.to_crs(basin_gdf.crs)
    station_proj = station_proj.to_crs(basin_gdf.crs)
    return gpd.sjoin(station_proj, basin_buffered, how="inner", predicate="within")


if __name__ == "__main__":
    basin_gdf = gpd.read_file(
        r"D:\Code\songliaodb_analysis\data\11rsvr_basins_shp\21100150-大伙房水库                    .shp"
    )
    station_gdf = gpd.read_file(
        r"D:\Code\songliaodb_analysis\results\chn_dllg_data\all_stations.shp"
    )
    stations = stations_within_basin(basin_gdf, station_gdf, buffer_m=5000)
    print(stations)
