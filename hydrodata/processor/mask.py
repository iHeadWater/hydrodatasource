#!/usr/bin/env python
# coding: utf-8

"""
该模块用于计算流域平均

- `mean_over_basin` - 计算流域平均

"""


import os
import numpy as np
import xarray as xr
import geopandas as gpd
import dask.array as da
import itertools
from shapely.geometry import Polygon
import hydrodata.configs.config as conf


def mean_over_basin(basin, basin_id, dataset, data_name, lon="lon", lat="lat"):
    """
    计算流域平均

    Todo:
        - 根据grid数据生成网格
        - 网格与流域相交
        - 以流域为单位计算流域内加权值


    Args:
        basin (GeoDataframe): 必选，流域的矢量数据，通过geopandas读取
        basin_id (str): 必选，表示流域编号的字段名称
        dataset (DataArray): 必选，表示grid数据集，通过xarray读取，只含有变量和经纬度
        data_name (str): 必选，表示grid数据集中需要参与计算的变量名称
        lon (str): 可选，grid数据集中经度坐标名称
        lat (str): 可选，grid数据集中纬度坐标名称

    Returns
        data (Dataframe): 流域编号和对应的平均值

    """

    grid = grid_to_gdf(dataset, data_name, lon=lon, lat=lat)

    intersects = gpd.overlay(grid, basin, how="intersection")
    intersects = intersects.to_crs(epsg=3857)
    intersects["Area"] = intersects.area
    intersects = intersects.to_crs(epsg=4326)

    return intersects.groupby(basin_id).apply(wavg, data_name, "Area")


def wavg(group, avg_name, weight_name):
    d = group[avg_name]
    w = group[weight_name]
    try:
        return (d * w).sum() / w.sum()
    except ZeroDivisionError:
        return d.mean()


def grid_to_gdf(dataset, data_name, lon, lat):
    lons = dataset[lon].values
    lats = dataset[lat].values
    delta = lons[1] - lons[0]

    geometry = []
    values = []
    HBlons = []
    HBlats = []

    delta_lon = lons.size
    delta_lat = lats.size

    for i, j in itertools.product(range(delta_lon), range(delta_lat)):
        HBLON = lons[i]
        HBLAT = lats[j]

        HBlons.append(HBLON)
        HBlats.append(HBLAT)

        geometry.append(
            Polygon(
                [
                    (HBLON - delta / 2, HBLAT + delta / 2),
                    (HBLON + delta / 2, HBLAT + delta / 2),
                    (HBLON + delta / 2, HBLAT - delta / 2),
                    (HBLON - delta / 2, HBLAT - delta / 2),
                ]
            )
        )

        try:
            values.append(float(dataset[data_name].isel(lon=i, lat=j).data))
        except Exception:
            values.append(float(dataset[data_name].isel(longitude=i, latitude=j).data))

    data = gpd.GeoDataFrame(crs="EPSG:4326", geometry=geometry)
    data["HBlon"] = HBlons
    data["HBlat"] = HBlats
    data[data_name] = values
    # data['geometry']=geometry

    return data


def gen_grids(bbox, resolution, offset):
    lx = bbox[0]
    rx = bbox[2]
    LLON = round(
        int(lx)
        + resolution * int((lx - int(lx)) / resolution + 0.5)
        + offset
        * (int(lx * 10) / 10 + offset - lx)
        / abs(int(lx * 10) / 10 + offset - lx),
        3,
    )
    RLON = round(
        int(rx)
        + resolution * int((rx - int(rx)) / resolution + 0.5)
        - offset
        * (int(rx * 10) / 10 + offset - rx)
        / abs(int(rx * 10) / 10 + offset - rx),
        3,
    )

    by = bbox[1]
    ty = bbox[3]
    BLAT = round(
        int(by)
        + resolution * int((by - int(by)) / resolution + 0.5)
        + offset
        * (int(by * 10) / 10 + offset - by)
        / abs(int(by * 10) / 10 + offset - by),
        3,
    )
    TLAT = round(
        int(ty)
        + resolution * int((ty - int(ty)) / resolution + 0.5)
        - offset
        * (int(ty * 10) / 10 + offset - ty)
        / abs(int(ty * 10) / 10 + offset - ty),
        3,
    )

    # print(LLON,BLAT,RLON,TLAT)

    xsize = round((RLON - LLON) / resolution) + 1
    ysize = round((TLAT - BLAT) / resolution) + 1

    # print(xsize, ysize)

    lons = np.linspace(LLON, RLON, xsize)
    lats = np.linspace(TLAT, BLAT, ysize)

    geometry = []
    HBlons = []
    HBlats = []

    for i in range(xsize):
        for j in range(ysize):
            HBLON = lons[i]
            HBLAT = lats[j]

            HBlons.append(HBLON)
            HBlats.append(HBLAT)

            geometry.append(
                Polygon(
                    [
                        (
                            round(HBLON - resolution / 2, 3),
                            round(HBLAT + resolution / 2, 3),
                        ),
                        (
                            round(HBLON + resolution / 2, 3),
                            round(HBLAT + resolution / 2, 3),
                        ),
                        (
                            round(HBLON + resolution / 2, 3),
                            round(HBLAT - resolution / 2, 3),
                        ),
                        (
                            round(HBLON - resolution / 2, 3),
                            round(HBLAT - resolution / 2, 3),
                        ),
                    ]
                )
            )

    data = gpd.GeoDataFrame(crs="EPSG:4326", geometry=geometry)
    data["lon"] = HBlons
    data["lat"] = HBlats

    return data


def get_para(data_name):
    if data_name.lower() in ["era5"]:
        return 0.1, 0
    elif data_name.lower() in ["gpm"]:
        return 0.1, 0.05
    elif data_name.lower() in ["gfs"]:
        return 0.25, 0
    else:
        raise Exception("未支持的数据产品")


def gen_mask(basin_id, watershed, dataname, save_dir="."):
    """
    计算流域平均

    Todo:
        - 根据grid数据生成网格
        - 网格与流域相交
        - 以流域为单位计算流域内加权值


    Args:
        watershed (GeoDataframe): 必选，流域的矢量数据，通过geopandas读取
        filedname (str): 必选，表示流域编号的字段名称
        dataname (DataArray): 必选，表示流域mask数据名称
        save_dir (str): 必选，表示流域mask文件生成路径

    Returns
        data (Dataframe): 流域编号和对应的平均值

    """

    for index, row in watershed.iterrows():
        # wid = row[filedname]
        wid = basin_id
        geo = row["geometry"]
        bbox = geo.bounds
        # print(geo.bounds)
        res, offset = get_para(dataname)

        grid = gen_grids(bbox, res, offset)
        grid = grid.to_crs(epsg=3857)
        grid["GRID_AREA"] = grid.area
        grid = grid.to_crs(epsg=4326)

        gs = gpd.GeoSeries.from_wkt([geo.wkt])
        sub = gpd.GeoDataFrame(crs="EPSG:4326", geometry=gs)

        intersects = gpd.overlay(grid, sub, how="intersection")
        intersects = intersects.to_crs(epsg=3857)
        intersects["BASIN_AREA"] = intersects.area
        intersects = intersects.to_crs(epsg=4326)
        intersects["w"] = intersects["BASIN_AREA"] / intersects["GRID_AREA"]

        grids = grid.set_index(["lon", "lat"]).join(
            intersects.set_index(["lon", "lat"]), lsuffix="_left", rsuffix="_right"
        )
        grids = grids.loc[:, ["w"]]
        grids.loc[grids.w.isnull(), "w"] = 0

        wds = grids.to_xarray()
        wds.to_netcdf(os.path.join(save_dir, f"mask-{wid}-{dataname}.nc"))


def gen_single_mask(basin_id, shp_path, dataname, mask_path, minio=False):
    if os.path.isfile(mask_path):
        return xr.open_dataset(mask_path)
    elif dataname in ["gpm", "gfs"]:
        if minio == False:
            shp_path = os.path.join(shp_path, basin_id, f"{basin_id}.shp")
            watershed = gpd.read_file(shp_path)
        else:
            watershed = gpd.read_file(conf.FS.open(shp_path))
        if not os.path.exists(mask_path):
            os.makedirs(mask_path)
        gen_mask(basin_id, watershed, dataname, save_dir=mask_path)
        mask_file_name = f"mask-{basin_id}-{dataname}.nc"
        mask_file_path = os.path.join(mask_path, mask_file_name)
        print(f"Mask file is generated in {mask_path}")
        return xr.open_dataset(mask_file_path)
    elif dataname != "merge":
        raise NotImplementedError("Only 'gpm', 'gfs' or 'merge' dataname is available.")


def mean_by_mask(src, var, mask):
    """
    计算流域平均

    Todo:
        - 根据grid数据生成网格mask
        - 根据mask提取数据
        - 以流域为单位计算流域内加权值

    Args:
        src (dataset): 必选，流域的网格数据，通过xarray读取
        var (str): 必选，表示网格数据中需要计算的变量名称
        mask (dataset): 必选，表示流域mask，通过xarray读取

    Returns
        m (float): 平均值

    """

    src_array = src[var].to_numpy()
    mask_array = mask["w"].to_numpy()

    src_array = da.from_array(src_array, chunks=(20, 10, 10))
    mask_array = da.from_array(mask_array, chunks=(10, 10))

    mask_array_expand = np.expand_dims(mask_array, 0).repeat(src_array.shape[0], 0)

    s = np.multiply(mask_array_expand, src_array)

    return np.nansum(s, axis=(1, 2)) / np.sum(mask_array)
