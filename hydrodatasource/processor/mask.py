#!/usr/bin/env python
# coding: utf-8

"""
该模块用于计算流域平均

- `mean_over_basin` - 计算流域平均

"""

import numpy as np
import xarray as xr
import geopandas as gpd
import pandas as pd
# import dask.array as da
import itertools
from geopandas import GeoDataFrame
from shapely.geometry import Polygon
import hydrodatasource.configs.config as hdscc


def mean_over_basin(basin, basin_id, dataset, data_name, lon="lon", lat="lat"):
    """
    计算流域平均

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
        / abs(int(rx * 10) / 10 + offset - rx + 0.00001),
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
    xsize = round((RLON - LLON) / resolution) + 1
    ysize = round((TLAT - BLAT) / resolution) + 1
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
    if data_name.lower() in ["era5_land"]:
        return 0.1, 0
    elif data_name.lower() in ["gpm"]:
        return 0.1, 0.05
    elif data_name.lower() in ["gfs", "era5"]:
        return 0.25, 0
    else:
        raise Exception("未支持的数据产品")


def gen_mask(watershed, dataname):
    """
    计算流域平均
    Args:
        watershed (GeoDataframe): 必选，流域的矢量数据，通过geopandas读取
        dataname (DataArray): 必选，表示流域mask数据名称
    Returns
        data (Dataframe): 流域编号和对应的平均值
    """
    for index, row in watershed.iterrows():
        # wid = row[filedname]
        # wid = basin_id
        geo = row["geometry"]
        bbox = geo.bounds
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
        # wds.to_netcdf(os.path.join(save_dir, f"mask-{wid}-{dataname}.nc"))
        return wds


def gen_mask_smap(smap_cell_array, basin_gdf):
    poly_list = []
    for i in range(0, smap_cell_array.shape[0] - 1):
        for j in range(0, smap_cell_array.shape[1] - 1):
            lon_half = (smap_cell_array[i][j + 1][0] - smap_cell_array[i][j][0]) / 2
            lat_half = (smap_cell_array[i][j][1] - smap_cell_array[i + 1][j][1]) / 2
            corner1 = (smap_cell_array[i][j][0] - lon_half, smap_cell_array[i][j][1] - lat_half)
            corner2 = (smap_cell_array[i][j][0] + lon_half, smap_cell_array[i][j][1] - lat_half)
            corner3 = (smap_cell_array[i][j][0] - lon_half, smap_cell_array[i][j][1] + lat_half)
            corner4 = (smap_cell_array[i][j][0] + lon_half, smap_cell_array[i][j][1] + lat_half)
            geom = Polygon((corner1, corner2, corner4, corner3, corner1))
            poly_list.append(geom)
    grid_gdf = GeoDataFrame(geometry=poly_list)
    # intersects = gpd.overlay(grid_gdf, basin_gdf, how="intersection")
    intersects = gpd.sjoin(grid_gdf, basin_gdf, how="inner")
    intersects["w"] = intersects.area / grid_gdf.geometry.area
    if len(intersects) < len(grid_gdf):
        zero_arr = np.append(np.array([intersects.iloc[len(intersects)-1].to_numpy()[:-2]], dtype=object), [np.nan, 0])
        concat_df = pd.DataFrame(np.repeat(zero_arr, len(grid_gdf)-len(intersects)).
                                 reshape((len(zero_arr), len(grid_gdf)-len(intersects))).T, columns=intersects.columns)
        intersects = pd.concat([intersects, concat_df], ignore_index=True)
    wds = intersects.drop(columns=['AREA', 'index_right', 'geometry']).to_xarray()
    return wds


def gen_single_mask(basin_id, watershed, dataname):
    mask_file_name = f"mask-{basin_id.rstrip('.zip')}-{dataname}.nc"
    s3_mask_path = f"s3://basins-origin/hour_data/1h/grid_data/{mask_file_name}"
    if hdscc.FS.exists(s3_mask_path):
        mask = xr.open_dataset(hdscc.FS.open(s3_mask_path))
    else:
        if dataname in ['gpm', 'gfs', 'era5_land', 'era5']:
            mask = gen_mask(watershed, dataname)
        elif dataname == 'smap':
            # w,e,n,s
            bounds_array = watershed.bounds.to_numpy()[0]
            bbox = [bounds_array[0], bounds_array[2], bounds_array[3], bounds_array[1]]
            # 不管什么时间，同一个bbox下，得到的smap数据和网格总是一致
            lon_array = np.load('smap_lon.npy')
            lat_array = np.load('smap_lat.npy')
            w_index = np.argwhere(lon_array >= bbox[0])[0][0]
            e_index = np.argwhere(lon_array <= bbox[1])[-1][0]
            n_index = np.argwhere(lat_array <= bbox[2])[0][0]
            s_index = np.argwhere(lat_array >= bbox[3])[-1][0]
            lon_slice = lon_array[w_index: e_index + 2]
            lat_slice = lat_array[n_index: s_index + 2]
            smap_cell_array = np.ndarray((lat_slice.shape[0], lon_slice.shape[0]), dtype=object)
            for lat in range(0, lat_slice.shape[0]):
                for lon in range(0, lon_slice.shape[0]):
                    smap_cell_array[lat, lon] = (lon_slice[lon], lat_slice[lat])
            mask = gen_mask_smap(smap_cell_array, watershed)
        else:
            mask = xr.Dataset()
        hdscc.FS.write_bytes(s3_mask_path, mask.to_netcdf())
    return mask


def mean_by_mask(src, var, mask):
    """
    计算流域平均

    Args:
        src (dataset): 必选，流域的网格数据，通过xarray读取
        var (str): 必选，表示网格数据中需要计算的变量名称
        mask (dataset): 必选，表示流域mask，通过xarray读取

    Returns
        m (float): 平均值
    """
    src_array = src[var].to_numpy()
    if 'tp' in src.data_vars:
        mask_array = mask["w"].to_numpy().T
        mask_array_expand = np.expand_dims(mask_array, 0).repeat(src_array.shape[0], 0)
    elif 'sm_surface' in src.data_vars:
        mask_array = mask["w"].to_numpy().reshape(src_array.shape)
        mask_array_expand = np.expand_dims(mask_array, 0)
    else:
        mask_array = mask["w"].to_numpy()
        mask_array_expand = np.expand_dims(mask_array, 0).repeat(src_array.shape[0], 0)
    # src_array = da.from_array(src_array, chunks='auto')
    # mask_array = da.from_array(mask_array, chunks='auto')
    s = np.multiply(mask_array_expand, src_array)
    return np.nansum(s, axis=(1, 2)) / np.sum(mask_array)
