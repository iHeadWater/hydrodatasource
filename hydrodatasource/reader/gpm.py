"""
Author: Jianfeng Zhu
Date: 2022-11-03 09:16:41
LastEditTime: 2025-01-02 18:56:35
LastEditors: Wenyu Ouyang
Description: 从minio中读取gpm数据
FilePath: \hydrodatasource\hydrodatasource\reader\gpm.py
Copyright (c) 2022-2025 Wenyu Ouyang. All rights reserved.
"""

import numpy as np
import xarray as xr
import calendar
import dask
import json

from ..configs.config import FS, RO
from ..utils.utils import regen_box


# 后期从minio读取
start = np.datetime64("2016-01-01T00:00:00.000000000")
end = np.datetime64("2023-08-17T23:30:00.000000000")
change = np.datetime64("2023-07-01T23:30:00.000000000")

box = (73.05, 3.05, 135.95, 53.95)

variables = [
    "HQobservationTime",
    "HQprecipSource",
    "HQprecipitation",
    "IRkalmanFilterWeight",
    "IRprecipitation",
    "precipitationCal",
    "precipitationUncal",
    "probabilityLiquidPrecipitation",
    "randomError",
]


dask.config.set({"array.slicing.split_large_chunks": False})


def get_dataset_year(bucket_name, start_time, end_time, bbox, time_chunks):
    year = str(start_time)[:4]

    chunks = {"time": time_chunks}
    ds = xr.open_dataset(
        "reference://",
        engine="zarr",
        chunks=chunks,
        backend_kwargs={
            "consolidated": False,
            "storage_options": {
                "fo": FS.open(
                    f"s3://{bucket_name}/geodata/gpm/{year}/gpm{year}_inc.json"
                ),
                "remote_protocol": "s3",
                "remote_options": RO,
            },
        },
    )

    ds = ds["precipitationCal"]
    # ds.to_dataframe().filter(['precipitationCal','precipitationUncal']).to_xarray()

    # ds = ds.rename({"longitude": "lon", "latitude": "lat"})
    ds = ds.transpose("time", "lon", "lat")

    if start_time < start:
        start_time = start

    if end_time > end:
        end_time = end

    times = slice(start_time, end_time)
    ds = ds.sel(time=times)

    bbox = regen_box(bbox, 0.1, 0.05)

    if bbox[0] < box[0]:
        left = box[0]
    else:
        left = bbox[0]

    if bbox[1] < box[1]:
        bottom = box[1]
    else:
        bottom = bbox[1]

    if bbox[2] > box[2]:
        right = box[2]
    else:
        right = bbox[2]

    if bbox[3] > box[3]:
        top = box[3]
    else:
        top = bbox[3]

    longitudes = slice(left - 0.00001, right + 0.00001)
    latitudes = slice(bottom - 0.00001, top + 0.00001)

    ds = ds.sortby("lat", ascending=True)
    ds = ds.sel(lon=longitudes, lat=latitudes)

    return ds


def get_dataset_month(bucket_name, start_time, end_time, bbox, time_chunks):
    year = str(start_time)[:4]
    month = str(start_time)[5:7].zfill(2)

    chunks = {"time": time_chunks}
    ds = xr.open_dataset(
        "reference://",
        engine="zarr",
        chunks=chunks,
        backend_kwargs={
            "consolidated": False,
            "storage_options": {
                "fo": FS.open(
                    f"s3://{bucket_name}/geodata/gpm/{year}/{month}/gpm{year}{month}_inc.json"
                ),
                "remote_protocol": "s3",
                "remote_options": RO,
            },
        },
    )

    ds = cf2datetime(ds)
    ds = ds["precipitationCal"]
    # ds.to_dataframe().filter(['precipitationCal','precipitationUncal']).to_xarray()

    # ds = ds.rename({"longitude": "lon", "latitude": "lat"})
    ds = ds.transpose("time", "lon", "lat")

    start_time = max(start_time, start)
    end_time = min(end_time, end)
    times = slice(start_time, end_time)
    ds = ds.sel(time=times)

    bbox = regen_box(bbox, 0.1, 0.05)

    if bbox[0] < box[0]:
        left = box[0]
    else:
        left = bbox[0]

    if bbox[1] < box[1]:
        bottom = box[1]
    else:
        bottom = bbox[1]

    if bbox[2] > box[2]:
        right = box[2]
    else:
        right = bbox[2]

    if bbox[3] > box[3]:
        top = box[3]
    else:
        top = bbox[3]

    longitudes = slice(left - 0.00001, right + 0.00001)
    latitudes = slice(bottom - 0.00001, top + 0.00001)

    ds = ds.sortby("lat", ascending=True)
    ds = ds.sel(lon=longitudes, lat=latitudes)

    return ds


def get_dataset_day(bucket_name, start_time, end_time, bbox, time_chunks):
    year = str(start_time)[:4]
    month = str(start_time)[5:7].zfill(2)
    day = str(end)[8:10].zfill(2)

    chunks = {"time": time_chunks}
    ds = xr.open_dataset(
        "reference://",
        engine="zarr",
        chunks=chunks,
        backend_kwargs={
            "consolidated": False,
            "storage_options": {
                "fo": FS.open(
                    f"s3://{bucket_name}/geodata/gpm/{year}/{month}/gpm{year}{month}_{day}.json"
                ),
                "remote_protocol": "s3",
                "remote_options": RO,
            },
        },
    )

    ds = cf2datetime(ds)
    ds = ds["precipitationCal"]
    # ds.to_dataframe().filter(['precipitationCal','precipitationUncal']).to_xarray()

    # ds = ds.rename({"longitude": "lon", "latitude": "lat"})
    ds = ds.transpose("time", "lon", "lat")

    if start_time < start:
        start_time = start

    if end_time > end:
        end_time = end

    times = slice(start_time, end_time)
    ds = ds.sel(time=times)

    bbox = regen_box(bbox, 0.1, 0.05)

    if bbox[0] < box[0]:
        left = box[0]
    else:
        left = bbox[0]

    if bbox[1] < box[1]:
        bottom = box[1]
    else:
        bottom = bbox[1]

    if bbox[2] > box[2]:
        right = box[2]
    else:
        right = bbox[2]

    if bbox[3] > box[3]:
        top = box[3]
    else:
        top = bbox[3]

    longitudes = slice(left - 0.00001, right + 0.00001)
    latitudes = slice(bottom - 0.00001, top + 0.00001)

    ds = ds.sortby("lat", ascending=True)
    ds = ds.sel(lon=longitudes, lat=latitudes)

    return ds


def cf2datetime(ds):
    ds = ds.copy()
    time_tmp1 = ds.indexes["time"]
    attrs = ds.coords["time"].attrs
    time_tmp2 = []
    for i in range(time_tmp1.shape[0]):
        tmp = time_tmp1[i]
        a = str(tmp.year).zfill(4)
        b = str(tmp.month).zfill(2)
        c = str(tmp.day).zfill(2)
        d = str(tmp.hour).zfill(2)
        e = str(tmp.minute).zfill(2)
        f = str(tmp.second).zfill(2)
        time_tmp2.append(
            np.datetime64("{}-{}-{} {}:{}:{}.00000000".format(a, b, c, d, e, f))
        )
    ds = ds.assign_coords(time=time_tmp2)
    ds.coords["time"].attrs = attrs

    return ds


def open_gpm_dataset(
    bucket_name,
    start_time=np.datetime64("2023-01-01T00:00:00.000000000"),
    end_time=np.datetime64("2023-01-02T00:00:00.000000000"),
    bbox=box,
    time_chunks=48,
):
    """
    从minio服务器读取gpm数据

    Args:
        start_time (datetime64): 开始时间
        end_time (datetime64): 结束时间
        bbox (list|tuple): 四至范围
        time_chunks (int): 分块数量

    Returns:
        dataset (Dataset): 读取结果
    """

    if end_time <= start_time:
        raise Exception("结束时间不能早于开始时间")

    if end_time <= change:
        # 早于20230701

        year_start = int(str(start_time)[:4])
        year_end = int(str(end_time)[:4])

        if year_start == year_end:
            ds = get_dataset_year(
                bucket_name,
                start_time=start_time,
                end_time=end_time,
                bbox=bbox,
                time_chunks=time_chunks,
            )
            return ds

        elif year_start < year_end:
            dss = []
            years = range(year_start, year_end + 1)
            for year in years:
                if year == year_start:
                    dss.append(
                        get_dataset_year(
                            bucket_name,
                            start_time=start_time,
                            end_time=np.datetime64(f"{year}-12-31T23:30:00.000000000"),
                            bbox=bbox,
                            time_chunks=time_chunks,
                        )
                    )

                elif year == year_end:
                    dss.append(
                        get_dataset_year(
                            bucket_name,
                            start_time=np.datetime64(
                                f"{year}-01-01T00:00:00.000000000"
                            ),
                            end_time=end_time,
                            bbox=bbox,
                            time_chunks=time_chunks,
                        )
                    )

                else:
                    dss.append(
                        get_dataset_year(
                            bucket_name,
                            start_time=np.datetime64(
                                f"{year}-01-01T00:00:00.000000000"
                            ),
                            end_time=np.datetime64(f"{year}-12-31T23:30:00.000000000"),
                            bbox=bbox,
                            time_chunks=time_chunks,
                        )
                    )
            return xr.merge(dss)

    elif change <= start_time:
        # 晚于20230701

        year_start = int(str(start_time)[:4])
        year_end = int(str(end_time)[:4])
        month_start = int(str(start_time)[5:7])
        month_end = int(str(end_time)[5:7])
        end_month = int(str(end)[5:7])

        if year_start == year_end:
            if month_end < end_month:
                if month_start == month_end:
                    ds = get_dataset_month(
                        bucket_name,
                        start_time=start_time,
                        end_time=end_time,
                        bbox=bbox,
                        time_chunks=time_chunks,
                    )
                    return ds

                else:
                    dss = []
                    for m in range(month_start, month_end + 1):
                        if m == month_start:
                            d = calendar.monthrange(year_start, m)[1]
                            dss.append(
                                get_dataset_month(
                                    bucket_name,
                                    start_time=start_time,
                                    end_time=np.datetime64(
                                        f"{year_start}-{str(m).zfill(2)}-{str(d).zfill(2)}T23:30:00.000000000"
                                    ),
                                    bbox=bbox,
                                    time_chunks=time_chunks,
                                )
                            )
                        elif m == month_end:
                            dss.append(
                                get_dataset_month(
                                    bucket_name,
                                    start_time=np.datetime64(
                                        f"{year_start}-{str(m).zfill(2)}-01T00:00:00.000000000"
                                    ),
                                    end_time=end_time,
                                    bbox=bbox,
                                    time_chunks=time_chunks,
                                )
                            )
                        else:
                            d = calendar.monthrange(year_start, m)[1]
                            dss.append(
                                get_dataset_month(
                                    bucket_name,
                                    start_time=np.datetime64(
                                        f"{year_start}-{str(m).zfill(2)}-01T00:00:00.000000000"
                                    ),
                                    end_time=np.datetime64(
                                        f"{year_start}-{str(m).zfill(2)}-{str(d).zfill(2)}T23:30:00.000000000"
                                    ),
                                    bbox=bbox,
                                    time_chunks=time_chunks,
                                )
                            )

                    return xr.merge(dss)

            else:
                if month_start == month_end:
                    ds = get_dataset_day(
                        bucket_name,
                        start_time=start_time,
                        end_time=end_time,
                        bbox=bbox,
                        time_chunks=time_chunks,
                    )
                    return ds

                else:
                    dss = []
                    for m in range(month_start, month_end + 1):
                        if m == month_start:
                            d = calendar.monthrange(year_start, m)[1]
                            dss.append(
                                get_dataset_month(
                                    bucket_name,
                                    start_time=start_time,
                                    end_time=np.datetime64(
                                        f"{year_start}-{str(m).zfill(2)}-{str(d).zfill(2)}T23:30:00.000000000"
                                    ),
                                    bbox=bbox,
                                    time_chunks=time_chunks,
                                )
                            )
                        elif m == month_end:
                            dss.append(
                                get_dataset_day(
                                    bucket_name,
                                    start_time=np.datetime64(
                                        f"{year_start}-{str(m).zfill(2)}-01T00:00:00.000000000"
                                    ),
                                    end_time=end_time,
                                    bbox=bbox,
                                    time_chunks=time_chunks,
                                )
                            )
                        else:
                            d = calendar.monthrange(year_start, m)[1]
                            dss.append(
                                get_dataset_month(
                                    bucket_name,
                                    start_time=np.datetime64(
                                        f"{year_start}-{str(m).zfill(2)}-01T00:00:00.000000000"
                                    ),
                                    end_time=np.datetime64(
                                        f"{year_start}-{str(m).zfill(2)}-{str(d).zfill(2)}T23:30:00.000000000"
                                    ),
                                    bbox=bbox,
                                    time_chunks=time_chunks,
                                )
                            )

                    return xr.merge(dss)

        else:
            dss = []

            for y in range(year_start, year_end):
                if y == year_start:
                    for m in range(month_start, 13):
                        if m == month_start:
                            d = calendar.monthrange(y, m)[1]
                            dss.append(
                                get_dataset_month(
                                    bucket_name,
                                    start_time=start_time,
                                    end_time=np.datetime64(
                                        f"{str(y).zfill(4)}-{str(m).zfill(2)}-{str(d).zfill(2)}T23:30:00.000000000"
                                    ),
                                    bbox=bbox,
                                    time_chunks=time_chunks,
                                )
                            )
                        else:
                            d = calendar.monthrange(y, m)[1]
                            dss.append(
                                get_dataset_month(
                                    bucket_name,
                                    start_time=np.datetime64(
                                        f"{str(y).zfill(4)}-{str(m).zfill(2)}-01T00:00:00.000000000"
                                    ),
                                    end_time=np.datetime64(
                                        f"{str(y).zfill(4)}-{str(m).zfill(2)}-{str(d).zfill(2)}T23:30:00.000000000"
                                    ),
                                    bbox=bbox,
                                    time_chunks=time_chunks,
                                )
                            )

                elif y == year_end:
                    for m in range(1, month_end + 1):
                        if m == end_month:
                            dss.append(
                                get_dataset_day(
                                    bucket_name,
                                    start_time=np.datetime64(
                                        f"{str(y).zfill(4)}-{str(m).zfill(2)}-01T00:00:00.000000000"
                                    ),
                                    end_time=end_time,
                                    bbox=bbox,
                                    time_chunks=time_chunks,
                                )
                            )
                        else:
                            dss.append(
                                get_dataset_month(
                                    bucket_name,
                                    start_time=np.datetime64(
                                        f"{str(y).zfill(4)}-{str(m).zfill(2)}-01T00:00:00.000000000"
                                    ),
                                    end_time=np.datetime64(
                                        f"{str(y).zfill(4)}-{str(m).zfill(2)}-{str(d).zfill(2)}T23:30:00.000000000"
                                    ),
                                    bbox=bbox,
                                    time_chunks=time_chunks,
                                )
                            )

                else:
                    for m in range(1, 13):
                        d = calendar.monthrange(y, m)[1]
                        dss.append(
                            get_dataset_month(
                                bucket_name,
                                start_time=np.datetime64(
                                    f"{str(y).zfill(4)}-{str(m).zfill(2)}-01T00:00:00.000000000"
                                ),
                                end_time=np.datetime64(
                                    f"{str(y).zfill(4)}-{str(m).zfill(2)}-{str(d).zfill(2)}T23:30:00.000000000"
                                ),
                                bbox=bbox,
                                time_chunks=time_chunks,
                            )
                        )

            return xr.merge(dss)

    else:
        # 包含20230701

        year_start = int(str(start_time)[:4])
        year_end = int(str(end_time)[:4])
        month_start = int(str(start_time)[5:7])
        month_end = int(str(end_time)[5:7])
        end_month = int(str(end)[5:7])

        dss = []

        for y in range(year_start, 2024):
            if y == year_start and y == 2023:
                dss.append(
                    get_dataset_year(
                        bucket_name,
                        start_time=start_time,
                        end_time=np.datetime64("2023-07-01T23:30:00.000000000"),
                        bbox=bbox,
                        time_chunks=time_chunks,
                    )
                )
            elif y == year_start and y < 2023:
                dss.append(
                    get_dataset_year(
                        bucket_name,
                        start_time=start_time,
                        end_time=np.datetime64("2023-12-31T23:30:00.000000000"),
                        bbox=bbox,
                        time_chunks=time_chunks,
                    )
                )
            elif y == 2023:
                dss.append(
                    get_dataset_year(
                        bucket_name,
                        start_time=np.datetime64("2023-01-01T00:00:00.000000000"),
                        end_time=np.datetime64("2023-07-01T23:30:00.000000000"),
                        bbox=bbox,
                        time_chunks=time_chunks,
                    )
                )
            else:
                dss.append(
                    get_dataset_year(
                        bucket_name,
                        start_time=np.datetime64("2023-01-01T00:00:00.000000000"),
                        end_time=np.datetime64("2023-12-31T23:30:00.000000000"),
                        bbox=bbox,
                        time_chunks=time_chunks,
                    )
                )

        for y in range(2023, year_end + 1):
            if y == year_end and y == 2023:
                for m in range(7, month_end + 1):
                    if m == month_end and m == 7:
                        dss.append(
                            get_dataset_month(
                                bucket_name,
                                start_time=np.datetime64(
                                    f"{str(y).zfill(4)}-{str(m).zfill(2)}-02T00:00:00.000000000"
                                ),
                                end_time=end_time,
                                bbox=bbox,
                                time_chunks=time_chunks,
                            )
                        )
                    elif m == month_end and m == end_month:
                        dss.append(
                            get_dataset_day(
                                bucket_name,
                                start_time=np.datetime64(
                                    f"{str(y).zfill(4)}-{str(m).zfill(2)}-01T00:00:00.000000000"
                                ),
                                end_time=end_time,
                                bbox=bbox,
                                time_chunks=time_chunks,
                            )
                        )
                    elif m == month_end and m > 7:
                        dss.append(
                            get_dataset_month(
                                bucket_name,
                                start_time=np.datetime64(
                                    f"{str(y).zfill(4)}-{str(m).zfill(2)}-01T00:00:00.000000000"
                                ),
                                end_time=end_time,
                                bbox=bbox,
                                time_chunks=time_chunks,
                            )
                        )
                    elif m == 7:
                        dss.append(
                            get_dataset_month(
                                bucket_name,
                                start_time=np.datetime64(
                                    "2023-07-02T00:00:00.000000000"
                                ),
                                end_time=np.datetime64("2023-07-31T23:30:00.000000000"),
                                bbox=bbox,
                                time_chunks=time_chunks,
                            )
                        )
                    else:
                        d = calendar.monthrange(y, m)[1]
                        dss.append(
                            get_dataset_year(
                                bucket_name,
                                start_time=np.datetime64(
                                    f"{str(y).zfill(4)}-{str(m).zfill(2)}-01T00:00:00.000000000"
                                ),
                                end_time=np.datetime64(
                                    f"{str(y).zfill(4)}-{str(m).zfill(2)}-{str(d).zfill(2)}T23:30:00.000000000"
                                ),
                                bbox=bbox,
                                time_chunks=time_chunks,
                            )
                        )
            elif y == year_end and y > 2023:
                for m in range(1, month_end + 1):
                    if m == month_end and m == end_month:
                        dss.append(
                            get_dataset_day(
                                bucket_name,
                                start_time=np.datetime64(
                                    f"{str(y).zfill(4)}-{str(m).zfill(2)}-01T00:00:00.000000000"
                                ),
                                end_time=end_time,
                                bbox=bbox,
                                time_chunks=time_chunks,
                            )
                        )
                    elif m == month_end:
                        dss.append(
                            get_dataset_month(
                                bucket_name,
                                start_time=np.datetime64(
                                    f"{str(y).zfill(4)}-{str(m).zfill(2)}-01T00:00:00.000000000"
                                ),
                                end_time=end_time,
                                bbox=bbox,
                                time_chunks=time_chunks,
                            )
                        )
                    else:
                        d = calendar.monthrange(y, m)[1]
                        dss.append(
                            get_dataset_year(
                                bucket_name,
                                start_time=np.datetime64(
                                    f"{str(y).zfill(4)}-{str(m).zfill(2)}-01T00:00:00.000000000"
                                ),
                                end_time=np.datetime64(
                                    f"{str(y).zfill(4)}-{str(m).zfill(2)}-{str(d).zfill(2)}T23:30:00.000000000"
                                ),
                                bbox=bbox,
                                time_chunks=time_chunks,
                            )
                        )
            elif y == 2023:
                for m in range(7, 13):
                    if m == 7:
                        dss.append(
                            get_dataset_year(
                                bucket_name,
                                start_time=np.datetime64(
                                    "2023-07-02T00:00:00.000000000"
                                ),
                                end_time=np.datetime64("2023-07-31T23:30:00.000000000"),
                                bbox=bbox,
                                time_chunks=time_chunks,
                            )
                        )
                    else:
                        d = calendar.monthrange(y, m)[1]
                        dss.append(
                            get_dataset_year(
                                bucket_name,
                                start_time=np.datetime64(
                                    f"{str(y).zfill(4)}-{str(m).zfill(2)}-01T00:00:00.000000000"
                                ),
                                end_time=np.datetime64(
                                    f"{str(y).zfill(4)}-{str(m).zfill(2)}-{str(d).zfill(2)}T23:30:00.000000000"
                                ),
                                bbox=bbox,
                                time_chunks=time_chunks,
                            )
                        )
            else:
                for m in range(1, 13):
                    d = calendar.monthrange(y, m)[1]
                    dss.append(
                        get_dataset_year(
                            bucket_name,
                            start_time=np.datetime64(
                                f"{str(y).zfill(4)}-{str(m).zfill(2)}-01T00:00:00.000000000"
                            ),
                            end_time=np.datetime64(
                                f"{str(y).zfill(4)}-{str(m).zfill(2)}-{str(d).zfill(2)}T23:30:00.000000000"
                            ),
                            bbox=bbox,
                            time_chunks=time_chunks,
                        )
                    )

        return xr.merge(dss)


import geopandas as gpd


def from_shp(
    start_time=np.datetime64("2023-01-01T00:00:00.000000000"),
    end_time=np.datetime64("2023-01-02T00:00:00.000000000"),
    shp=None,
    time_chunks=48,
):
    """
    通过已有的矢量数据范围从minio服务器读取gpm数据

    Args:
        start_time (datetime64): 开始时间
        end_time (datetime64): 结束时间
        shp (str): 矢量数据路径
        time_chunks (int): 分块数量

    Returns:
        dataset (Dataset): 读取结果
    """

    gdf = gpd.GeoDataFrame.from_file(shp)
    b = gdf.bounds
    bbox = regen_box(
        (b.loc[0]["minx"], b.loc[0]["miny"], b.loc[0]["maxx"], b.loc[0]["maxy"]),
        0.1,
        0.05,
    )

    ds = open_gpm_dataset(start_time, end_time, bbox, time_chunks)

    return ds


def from_aoi(
    start_time=np.datetime64("2023-01-01T00:00:00.000000000"),
    end_time=np.datetime64("2023-01-02T00:00:00.000000000"),
    aoi: gpd.GeoDataFrame = None,
    time_chunks=48,
):
    """
    用过已有的GeoPandas.GeoDataFrame对象从minio服务器读取gpm数据

    Args:
        start_time (datetime64): 开始时间
        end_time (datetime64): 结束时间
        aoi (GeoDataFrame): 已有的GeoPandas.GeoDataFrame对象
        time_chunks (int): 分块数量

    Returns:
        dataset (Dataset): 读取结果
    """

    b = aoi.bounds
    bbox = regen_box(
        (b.loc[0]["minx"], b.loc[0]["miny"], b.loc[0]["maxx"], b.loc[0]["maxy"]),
        0.1,
        0.05,
    )

    ds = open_gpm_dataset(start_time, end_time, bbox, time_chunks)

    return ds


if __name__ == "__main__":
    pass


class GPMReader:
    """
    用于从minio中读取gpm数据

    Methods:
        open_dataset(start_time, end_time, dataset, bbox, time_resolution): 从minio中读取gpm数据
        from_shp(start_time, end_time, dataset, shp, time_resolution): 通过已有的矢量数据范围从minio服务器读取gpm数据
        from_aoi(start_time, end_time, dataset, aoi, time_resolution): 用过已有的GeoPandas.GeoDataFrame对象从minio服务器读取gpm数据
    """

    def __init__(self):
        self._bucket_name = "test"

    def _get_dataset(self, scale, start_time, end_time, bbox, time_chunks):
        year = str(start_time)[:4]
        month = str(start_time)[5:7].zfill(2)
        day = str(self._endtime)[8:10].zfill(2)

        if scale == "Y":
            minio_path = f"s3://{self._bucket_name}/{self._dataset}/gpm{self._time_resolution}/{year}/gpm{year}_inc.json"

        elif scale == "M":
            minio_path = f"s3://{self._bucket_name}/{self._dataset}/gpm{self._time_resolution}/{year}/{month}/gpm{year}{month}_inc.json"

        chunks = {"time": time_chunks}
        ds = xr.open_dataset(
            "reference://",
            engine="zarr",
            chunks=chunks,
            backend_kwargs={
                "consolidated": False,
                "storage_options": {
                    "fo": minio_path,
                    "target_protocol": "s3",
                    "target_options": RO,
                    "remote_protocol": "s3",
                    "remote_options": RO,
                },
            },
        )

        # if self._time_resolution == '1d':
        #     ds = cf2datetime(ds)

        ds = ds["precipitationCal"]
        # ds.to_dataframe().filter(['precipitationCal','precipitationUncal']).to_xarray()

        # ds = ds.rename({"longitude": "lon", "latitude": "lat"})
        ds = ds.transpose("time", "lon", "lat")

        times = slice(start_time, end_time)
        ds = ds.sel(time=times)

        left = bbox[0]
        right = bbox[2]
        bottom = bbox[1]
        top = bbox[3]

        longitudes = slice(left - 0.00001, right + 0.00001)
        latitudes = slice(bottom - 0.00001, top + 0.00001)

        ds = ds.sortby("lat", ascending=True)
        ds = ds.sel(lon=longitudes, lat=latitudes)

        return ds

    def open_dataset(
        self,
        start_time=np.datetime64("2023-01-01T00:00:00.000000000"),
        end_time=np.datetime64("2023-01-02T00:00:00.000000000"),
        dataset="wis",
        bbox=(121, 39, 122, 40),
        time_resolution="1d",
        time_chunks=48,
    ):
        """
        从minio服务器读取gpm数据

        Args:
            start_time (datetime64): 开始时间
            end_time (datetime64): 结束时间
            dataset (str): wis或camels
            bbox (list|tuple): 四至范围
            time_resolution (str): 1d或30m
            time_chunks (int): 分块数量

        Returns:
            dataset (Dataset): 读取结果
        """

        if end_time <= start_time:
            raise Exception("结束时间不能早于开始时间")

        if bbox[0] > bbox[2] or bbox[1] > bbox[3]:
            raise Exception("四至范围错误")

        if dataset != "wis" and dataset != "camels":
            raise Exception("dataset参数错误")

        if time_resolution != "1d" and time_resolution != "30m":
            raise Exception("time_resolution参数错误")

        # dataset_name = get_dataset_name()
        if dataset == "wis":
            self._dataset = "geodata"
        elif dataset == "camels":
            self._dataset = "camdata"

        if time_resolution == "1d":
            self._time_resolution = "1d"
        elif time_resolution == "30m":
            self._time_resolution = ""

        with FS.open(
            os.path.join(
                self._bucket_name,
                f"{self._dataset}/gpm{self._time_resolution}/gpm{self._time_resolution}.json",
            )
        ) as f:
            cont = json.load(f)
            self._starttime = np.datetime64(cont["start"])
            self._endtime = np.datetime64(cont["end"])
            self._bbox = cont["bbox"]

        if start_time < self._starttime:
            start_time = self._starttime

        if end_time > self._endtime:
            end_time = self._endtime

        bbox = regen_box(bbox, 0.1, 0.05)

        if bbox[0] < self._bbox[0]:
            bbox[0] = self._bbox[0]
        if bbox[1] < self._bbox[1]:
            bbox[1] = self._bbox[1]
        if bbox[2] > self._bbox[2]:
            bbox[2] = self._bbox[2]
        if bbox[3] > self._bbox[3]:
            bbox[3] = self._bbox[3]

        year_start = int(str(start_time)[:4])
        year_end = int(str(end_time)[:4])
        end_year = int(str(self._endtime)[:4])

        if year_end < end_year:
            if year_start == year_end:
                ds = self._get_dataset(
                    scale="Y",
                    start_time=start_time,
                    end_time=end_time,
                    bbox=bbox,
                    time_chunks=time_chunks,
                )
                return ds

            elif year_start < year_end:
                dss = []
                years = range(year_start, year_end + 1)
                for year in years:
                    if year == year_start:
                        dss.append(
                            self._get_dataset(
                                scale="Y",
                                start_time=start_time,
                                end_time=np.datetime64(
                                    f"{year}-12-31T23:30:00.000000000"
                                ),
                                bbox=bbox,
                                time_chunks=time_chunks,
                            )
                        )

                    elif year == year_end:
                        dss.append(
                            self._get_dataset(
                                scale="Y",
                                start_time=np.datetime64(
                                    f"{year}-01-01T00:00:00.000000000"
                                ),
                                end_time=end_time,
                                bbox=bbox,
                                time_chunks=time_chunks,
                            )
                        )

                    else:
                        dss.append(
                            self._get_dataset(
                                scale="Y",
                                start_time=np.datetime64(
                                    f"{year}-01-01T00:00:00.000000000"
                                ),
                                end_time=np.datetime64(
                                    f"{year}-12-31T23:30:00.000000000"
                                ),
                                bbox=bbox,
                                time_chunks=time_chunks,
                            )
                        )
                return xr.merge(dss)

        else:
            month_end = int(str(end_time)[5:7])
            end_month = int(str(self._endtime)[5:7])

            if year_start == year_end:
                month_start = int(str(start_time)[5:7])
                if month_start == month_end:
                    return self._get_dataset(
                        scale="M",
                        start_time=start_time,
                        end_time=end_time,
                        bbox=bbox,
                        time_chunks=time_chunks,
                    )
                dss = []
                for m in range(month_start, month_end + 1):
                    if m == month_start:
                        d = calendar.monthrange(year_start, m)[1]
                        dss.append(
                            self._get_dataset(
                                scale="M",
                                start_time=start_time,
                                end_time=np.datetime64(
                                    f"{year_start}-{str(m).zfill(2)}-{str(d).zfill(2)}T23:30:00.000000000"
                                ),
                                bbox=bbox,
                                time_chunks=time_chunks,
                            )
                        )
                    elif m == month_end:
                        dss.append(
                            self._get_dataset(
                                scale="M",
                                start_time=np.datetime64(
                                    f"{year_start}-{str(m).zfill(2)}-01T00:00:00.000000000"
                                ),
                                end_time=end_time,
                                bbox=bbox,
                                time_chunks=time_chunks,
                            )
                        )
                    else:
                        d = calendar.monthrange(year_start, m)[1]
                        dss.append(
                            self._get_dataset(
                                scale="M",
                                start_time=np.datetime64(
                                    f"{year_start}-{str(m).zfill(2)}-01T00:00:00.000000000"
                                ),
                                end_time=np.datetime64(
                                    f"{year_start}-{str(m).zfill(2)}-{str(d).zfill(2)}T23:30:00.000000000"
                                ),
                                bbox=bbox,
                                time_chunks=time_chunks,
                            )
                        )

            else:
                dss = []

                for y in range(year_start, year_end + 1):
                    if y == year_start:
                        dss.append(
                            self._get_dataset(
                                scale="Y",
                                start_time=start_time,
                                end_time=np.datetime64(f"{y}-12-31T23:30:00.000000000"),
                                bbox=bbox,
                                time_chunks=time_chunks,
                            )
                        )

                    elif y == year_end:
                        for m in range(1, month_end + 1):
                            if m == month_end:
                                dss.append(
                                    self._get_dataset(
                                        scale="M",
                                        start_time=np.datetime64(
                                            f"{str(y).zfill(4)}-{str(m).zfill(2)}-01T00:00:00.000000000"
                                        ),
                                        end_time=end_time,
                                        bbox=bbox,
                                        time_chunks=time_chunks,
                                    )
                                )
                            else:
                                d = calendar.monthrange(y, m)[1]
                                dss.append(
                                    self._get_dataset(
                                        scale="M",
                                        start_time=np.datetime64(
                                            f"{str(y).zfill(4)}-{str(m).zfill(2)}-01T00:00:00.000000000"
                                        ),
                                        end_time=np.datetime64(
                                            f"{str(y).zfill(4)}-{str(m).zfill(2)}-{str(d).zfill(2)}T23:30:00.000000000"
                                        ),
                                        bbox=bbox,
                                        time_chunks=time_chunks,
                                    )
                                )

                    else:
                        dss.append(
                            self._get_dataset(
                                scale="Y",
                                start_time=np.datetime64(
                                    f"{y}-01-01T00:00:00.000000000"
                                ),
                                end_time=np.datetime64(f"{y}-12-31T23:30:00.000000000"),
                                bbox=bbox,
                                time_chunks=time_chunks,
                            )
                        )

            return xr.merge(dss)

    def from_shp(
        self,
        start_time=np.datetime64("2023-01-01T00:00:00.000000000"),
        end_time=np.datetime64("2023-01-02T00:00:00.000000000"),
        dataset="wis",
        shp=None,
        time_resolution="1d",
        time_chunks=48,
    ):
        """
        通过已有的矢量数据范围从minio服务器读取gpm数据

        Args:
            start_time (datetime64): 开始时间
            end_time (datetime64): 结束时间
            dataset (str): wis或camels
            shp (str): 矢量数据路径
            time_resolution (str): 1d或30m
            time_chunks (int): 分块数量

        Returns:
            dataset (Dataset): 读取结果
        """

        gdf = gpd.GeoDataFrame.from_file(shp)
        b = gdf.bounds
        bbox = regen_box(
            (b.loc[0]["minx"], b.loc[0]["miny"], b.loc[0]["maxx"], b.loc[0]["maxy"]),
            0.1,
            0.05,
        )

        ds = self.open_dataset(
            start_time, end_time, dataset, bbox, time_resolution, time_chunks
        )
        return ds

    def from_aoi(
        self,
        start_time=np.datetime64("2023-01-01T00:00:00.000000000"),
        end_time=np.datetime64("2023-01-02T00:00:00.000000000"),
        dataset="wis",
        aoi: gpd.GeoDataFrame = None,
        time_resolution="1d",
        time_chunks=48,
    ):
        """
        用过已有的GeoPandas.GeoDataFrame对象从minio服务器读取gpm数据

        Args:
            start_time (datetime64): 开始时间
            end_time (datetime64): 结束时间
            dataset (str): wis或camels
            aoi (GeoDataFrame): 已有的GeoPandas.GeoDataFrame对象
            time_resolution (str): 1d或30m
            time_chunks (int): 分块数量

        Returns:
            dataset (Dataset): 读取结果
        """

        b = aoi.bounds
        bbox = regen_box(
            (b.loc[0]["minx"], b.loc[0]["miny"], b.loc[0]["maxx"], b.loc[0]["maxy"]),
            0.1,
            0.05,
        )
        ds = self.open_dataset(
            start_time, end_time, dataset, bbox, time_resolution, time_chunks
        )
        return ds
