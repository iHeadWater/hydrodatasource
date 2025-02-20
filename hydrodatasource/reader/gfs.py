"""
Author: Jianfeng Zhu
Date: 2022-11-03 09:16:41
LastEditTime: 2025-01-02 18:19:32
LastEditors: Wenyu Ouyang
Description: 从minio中读取gfs数据
FilePath: \hydrodatasource\hydrodatasource\reader\gfs.py
Copyright (c) 2022-2025 Wenyu Ouyang. All rights reserved.
"""

import os
import numpy as np
import xarray as xr
import json
import geopandas as gpd

from ..configs.config import FS, RO
from ..utils.utils import regen_box


# 后期从minio读取
start = np.datetime64("2016-07-10")
end = np.datetime64("2023-08-17")
change = np.datetime64("2022-09-01")
# change_date = date(2022,9,1)
box = (115, 38, 136, 54)


# 2t  ----- temperature_2m_above_ground
# 2sh ----- specific_humidity_2m_above_ground
# 2r  ----- relative_humidity_2m_above_ground
# 10u ----- u_component_of_wind_10m_above_ground
# 10v ----- v_component_of_wind_10m_above_ground
# tp  ----- total_precipitation_surface
# pwat----- precipitable_water_entire_atmosphere
# tcc ----- total_cloud_cover_entire_atmosphere
# dswrf ----- downward_shortwave_radiation_flux

variables = {
    "dswrf": "downward_shortwave_radiation_flux",
    "pwat": "precipitable_water_entire_atmosphere",
    "2r": "relative_humidity_2m_above_ground",
    "2sh": "specific_humidity_2m_above_ground",
    "2t": "temperature_2m_above_ground",
    "tcc": "total_cloud_cover_entire_atmosphere",
    "tp": "total_precipitation_surface",
    "10u": "u_component_of_wind_10m_above_ground",
    "10v": "v_component_of_wind_10m_above_ground",
}


def open_gfs_dataset(
    bucket_name,
    data_variable="tp",
    creation_date=np.datetime64("2022-09-01"),
    creation_time="00",
    bbox=box,
    time_chunks=24,
):
    """
    从minio服务器读取gfs数据

    Args:
        data_variables (str): 数据变量，目前只支持tp，即降雨
        creation_date (datetime64): 创建日期
        creation_time (datetime64): 创建时间，即00\06\12\18之一
        bbox (list|tuple): 四至范围
        time_chunks (int): 分块数量

    Returns:
        dataset (Dataset): 读取结果
    """

    if data_variable in variables.keys():
        short_name = data_variable
        full_name = variables[data_variable]

        with FS.open(f"{bucket_name}/geodata/gfs/gfs.json") as f:
            cont = json.load(f)
            start = np.datetime64(cont[short_name][0]["start"])
            end = np.datetime64(cont[short_name][-1]["end"])

        if creation_date < start or creation_date > end:
            print("超出时间范围！")
            return

        if creation_time not in ["00", "06", "12", "18"]:
            print("creation_time必须是00、06、12、18之一！")
            return

        year = str(creation_date.astype("object").year)
        month = str(creation_date.astype("object").month).zfill(2)
        day = str(creation_date.astype("object").day).zfill(2)

        if creation_date < change:
            json_url = f"s3://{bucket_name}/geodata/gfs/gfs_history/{year}/{month}/{day}/gfs{year}{month}{day}.t{creation_time}z.0p25.json"
        else:
            json_url = f"s3://{bucket_name}/geodata/gfs/{short_name}/{year}/{month}/{day}/gfs{year}{month}{day}.t{creation_time}z.0p25.json"

        chunks = {"valid_time": time_chunks}
        ds = xr.open_dataset(
            "reference://",
            engine="zarr",
            chunks=chunks,
            backend_kwargs={
                "consolidated": False,
                "storage_options": {
                    "fo": FS.open(json_url),
                    "remote_protocol": "s3",
                    "remote_options": RO,
                },
            },
        )

        if creation_date < change:
            ds = ds[full_name]

        # ds = ds.filter_by_attrs(long_name=lambda v: v in data_variables)
        ds = ds.rename({"longitude": "lon", "latitude": "lat"})
        # ds = ds.transpose('time','valid_time','lon','lat')

        bbox = regen_box(bbox, 0.25, 0)

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

    else:
        print("变量名不存在！")


def from_shp(
    data_variable="tp",
    creation_date=np.datetime64("2022-09-01"),
    creation_time="00",
    shp=None,
    time_chunks=24,
):
    """
    通过已有的矢量数据范围从minio服务器读取gfs数据

    Args:
        data_variables (str): 数据变量，目前只支持tp，即降雨
        creation_date (datetime64): 创建日期
        creation_time (datetime64): 创建时间，即00\06\12\18之一
        shp (str): 矢量数据路径
        time_chunks (int): 分块数量

    Returns:
        dataset (Dataset): 读取结果
    """

    gdf = gpd.GeoDataFrame.from_file(shp)
    b = gdf.bounds
    bbox = regen_box(
        (b.loc[0]["minx"], b.loc[0]["miny"], b.loc[0]["maxx"], b.loc[0]["maxy"]), 0.1, 0
    )

    return open_gfs_dataset(
        data_variable, creation_date, creation_time, bbox, time_chunks
    )


def from_aoi(
    data_variable="tp",
    creation_date=np.datetime64("2022-09-01"),
    creation_time="00",
    aoi: gpd.GeoDataFrame = None,
    time_chunks=24,
):
    """
    通过已有的GeoPandas.GeoDataFrame对象从minio服务器读取gfs数据

    Args:
        data_variables (str): 数据变量，目前只支持tp，即降雨
        creation_date (datetime64): 创建日期
        creation_time (datetime64): 创建时间，即00\06\12\18之一
        aoi (GeoDataFrame): 已有的GeoPandas.GeoDataFrame对象
        time_chunks (int): 分块数量

    Returns:
        dataset (Dataset): 读取结果
    """
    b = aoi.bounds
    bbox = regen_box(
        (b.loc[0]["minx"], b.loc[0]["miny"], b.loc[0]["maxx"], b.loc[0]["maxy"]), 0.1, 0
    )

    return open_gfs_dataset(
        data_variable, creation_date, creation_time, bbox, time_chunks
    )


class GFSReader:
    """
    用于从minio中读取gpm数据

    Attributes:
        variables (dict): 变量名称及缩写

    Methods:
        open_dataset(data_variables, creation_date, creation_time, bbox): 从minio中读取gfs数据
        from_shp(data_variables, creation_date, creation_time, shp): 通过已有的矢量数据范围从minio服务器读取gfs数据
        from_aoi(data_variables, creation_date, creation_time, aoi): 用过已有的GeoPandas.GeoDataFrame对象从minio服务器读取gfs数据
    """

    def __init__(self):
        self._variables = {
            "dswrf": "downward_shortwave_radiation_flux",
            "pwat": "precipitable_water_entire_atmosphere",
            "2r": "relative_humidity_2m_above_ground",
            "2sh": "specific_humidity_2m_above_ground",
            "2t": "temperature_2m_above_ground",
            "tcc": "total_cloud_cover_entire_atmosphere",
            "tp": "total_precipitation_surface",
            "10u": "u_component_of_wind_10m_above_ground",
            "10v": "v_component_of_wind_10m_above_ground",
        }
        self._bucket_name = "test"
        self._default = "tp"

    @property
    def variables(self):
        return self._variables

    @property
    def default_variable(self):
        return self._default

    def set_default_variable(self, short_name):
        if short_name in self._variables.keys():
            self._default = short_name
        else:
            raise Exception("变量设置错误")

    def open_dataset(
        self,
        creation_date=np.datetime64("2022-09-01"),
        creation_time="00",
        dataset="wis",
        bbox=(115, 38, 136, 54),
        time_chunks=24,
    ):
        """
        从minio服务器读取gfs数据

        Args:
            creation_date (datetime64): 创建日期
            creation_time (datetime64): 创建时间，即00\06\12\18之一
            dataset (str): wis或camels
            bbox (list|tuple): 四至范围
            time_chunks (int): 分块数量

        Returns:
            dataset (Dataset): 读取结果
        """

        if bbox[0] > bbox[2] or bbox[1] > bbox[3]:
            raise Exception("四至范围格式错误")

        if dataset != "wis" and dataset != "camels":
            raise Exception("dataset参数错误")

        if dataset == "wis":
            self._dataset = "geodata"
        elif dataset == "camels":
            self._dataset = "camdata"

        with FS.open(
            os.path.join(self._bucket_name, f"{self._dataset}/gfs/gfs.json")
        ) as f:
            cont = json.load(f)
            self._paras = cont

        short_name = self._default
        full_name = self._variables[short_name]

        start = np.datetime64(self._paras[short_name][0]["start"])
        end = np.datetime64(self._paras[short_name][-1]["end"])

        if creation_date < start or creation_date > end:
            print("超出时间范围！")
            return

        if creation_time not in ["00", "06", "12", "18"]:
            print("creation_time必须是00、06、12、18之一！")
            return

        year = str(creation_date.astype("object").year)
        month = str(creation_date.astype("object").month).zfill(2)
        day = str(creation_date.astype("object").day).zfill(2)

        change = np.datetime64("2022-09-01")
        if creation_date < change:
            json_url = f"s3://{self._bucket_name}/{self._dataset}/gfs/gfs_history/{year}/{month}/{day}/gfs{year}{month}{day}.t{creation_time}z.0p25.json"
        else:
            json_url = f"s3://{self._bucket_name}/{self._dataset}/gfs/{short_name}/{year}/{month}/{day}/gfs{year}{month}{day}.t{creation_time}z.0p25.json"

        chunks = {"valid_time": time_chunks}
        ds = xr.open_dataset(
            "reference://",
            engine="zarr",
            chunks=chunks,
            backend_kwargs={
                "consolidated": False,
                "storage_options": {
                    "fo": json_url,
                    "target_protocol": "s3",
                    "target_options": RO,
                    "remote_protocol": "s3",
                    "remote_options": RO,
                },
            },
        )

        if creation_date < change:
            ds = ds[full_name]
            box = self._paras[short_name][0]["bbox"]
        else:
            box = self.paras[short_name][-1]["bbox"]

        # ds = ds.filter_by_attrs(long_name=lambda v: v in data_variables)
        ds = ds.rename({"longitude": "lon", "latitude": "lat"})
        # ds = ds.transpose('time','valid_time','lon','lat')

        bbox = regen_box(bbox, 0.25, 0)

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

    def from_shp(
        self,
        creation_date=np.datetime64("2022-09-01"),
        creation_time="00",
        dataset="wis",
        shp=None,
        time_chunks=24,
    ):
        """
        通过已有的矢量数据范围从minio服务器读取gfs数据

        Args:
            creation_date (datetime64): 创建日期
            creation_time (datetime64): 创建时间，即00\06\12\18之一
            dataset (str): wis或camels
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
            0,
        )

        ds = self.open_dataset(creation_date, creation_time, dataset, bbox, time_chunks)

        return ds

    def from_aoi(
        self,
        creation_date=np.datetime64("2022-09-01"),
        creation_time="00",
        dataset="wis",
        aoi: gpd.GeoDataFrame = None,
        time_chunks=24,
    ):
        """
        通过已有的GeoPandas.GeoDataFrame对象从minio服务器读取gfs数据

        Args:
            creation_date (datetime64): 创建日期
            creation_time (datetime64): 创建时间，即00\06\12\18之一
            dataset (str): wis或camels
            aoi (GeoDataFrame): 已有的GeoPandas.GeoDataFrame对象
            time_chunks (int): 分块数量

        Returns:
            dataset (Dataset): 读取结果
        """
        b = aoi.bounds
        bbox = regen_box(
            (b.loc[0]["minx"], b.loc[0]["miny"], b.loc[0]["maxx"], b.loc[0]["maxy"]),
            0.1,
            0,
        )

        ds = self.open_dataset(creation_date, creation_time, dataset, bbox, time_chunks)

        return ds
