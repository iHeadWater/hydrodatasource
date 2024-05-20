"""
Author: Wenyu Ouyang
Date: 2023-10-31 09:26:31
LastEditTime: 2024-03-28 08:33:33
LastEditors: Wenyu Ouyang
Description: Interface for reader
FilePath: \hydrodatasource\hydrodatasource\reader\reader.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

from abc import ABC, abstractmethod
import os
import json
import datetime
import xarray as xr

from hydrodatasource.processor.gpm_gfs import make1nc41basin
from hydrodatasource.configs.config import LOCAL_DATA_PATH, GRID_INTERIM_BUCKET, RO, FS
from hydrodatasource.processor.mask import gen_single_mask


class AOI:
    def __init__(self, aoi_type, aoi_param):
        self._aoi_type = aoi_type  # can be "grid", "station", "basin" etc.
        self._aoi_param = aoi_param  # this can be a bounding box, coordinates, etc.

    @property
    def aoi_type(self):
        return self._aoi_type

    @property
    def aoi_param(self):
        return self._aoi_param

    def get_mask(self):
        # If it's a polygon, return its mask for data extraction
        if self._aoi_type == "basin":
            return gen_single_mask(
                basin_id=self._aoi_param,
                shp_path=os.path.join(LOCAL_DATA_PATH, "datasets-origin", "shp"),
                dataname="gpm",
                mask_path=os.path.join(LOCAL_DATA_PATH, "datasets-origin", "mask"),
            )
        else:
            # basin only for now
            raise NotImplementedError("Only basin available for now")

    # ... any other useful methods to describe or manipulate the AOI


class CommonHandler(AOI):
    def __init__(self, aoi_type, aoi_param, region=None, time_periods=None):
        super().__init__(aoi_type, aoi_param)
        self._region = region  # camels or wis, "camels" is on behalf of us, "wis" is on behalf of cn
        self._time_periods = time_periods

    @property
    def region(self):
        return self._region

    @property
    def time_periods(self):
        return self._time_periods

    def is_valid_time_periods(self):
        # 检查 time_periods 是否是列表
        if not isinstance(self._time_periods, list):
            return False

        for period in self._time_periods:
            # 检查每个元素是否是包含两个元素的列表
            if not isinstance(period, list) or len(period) != 2:
                return False

            # 检查每个元素是否是字符串且符合日期时间格式
            for date_str in period:
                if not isinstance(date_str, str):
                    return False
                try:
                    datetime.datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S")
                except ValueError:
                    return False

        return True

    def read_file_from_minio(self):
        try:
            json_file_path = os.path.join(self._aoi_param, f"{self._dataname}.json")
            # 从 MinIO 读取 JSON 文件
            with FS.open(f"{GRID_INTERIM_BUCKET}/{json_file_path}") as f:
                json_data = json.load(f)
            # 使用 xarray 和 kerchunk 读取数据
            return xr.open_dataset(
                "reference://",
                engine="zarr",
                backend_kwargs={
                    "consolidated": False,
                    "storage_options": {
                        "fo": json_data,
                        "remote_protocol": "s3",
                        "remote_options": RO,
                    },
                },
            )
        except:
            raise FileNotFoundError(
                "Please check the file in the minio server. \
                This error is generally caused by the following two situations:\n\
                    1. There is no file for this basin in the minio server. Please make a new one and upload it.\n\
                    2. Check your settings. Make sure you have enough permission to access the minio server and the bucket."
            )


class DataHandler(CommonHandler):
    def __init__(
        self,
        aoi_type,
        aoi_param,
        region=None,
        time_periods=None,
        dataname=None,
        minio_read=True,
        local_save=True,
        minio_upload=False,
    ):
        super().__init__(aoi_type, aoi_param, region, time_periods)
        self._dataname = dataname
        self._minio_read = minio_read
        self._local_save = local_save
        self._minio_upload = minio_upload

    @property
    def dataname(self):
        return self._local_file_path

    @property
    def dataname(self):
        return self._dataname

    @property
    def minio_read(self):
        return self._minio_read

    @property
    def local_save(self):
        return self._local_save

    @property
    def minio_upload(self):
        return self._minio_upload

    def handle(self):
        if self._minio_read == True:
            return self.read_file_from_minio()
        else:
            return make1nc41basin(
                basin_id=self._aoi_param,
                dataname=self._dataname,
                local_path=LOCAL_DATA_PATH,
                mask_path=os.path.join(LOCAL_DATA_PATH, "datasets-origin", "mask"),
                shp_path=os.path.join(LOCAL_DATA_PATH, "datasets-origin", "shp"),
                dataset=self._region,
                time_periods=self._time_periods,
                local_save=self._local_save,
                minio_upload=self._minio_upload,
            )

    # TODO: time_periods is only used in make1nc41basin funtion, it needs to be used in other cases


class DataReaderStrategy(ABC):
    @abstractmethod
    def read(self, path: str, aoi: AOI):
        pass


class AbstractFileReader(DataReaderStrategy):
    def __init__(self, data_handler):
        self.data_handler = data_handler

    @abstractmethod
    def configure(self, path: str, aoi: AOI):
        pass

    def read(self, path: str, aoi: AOI):
        configuration = self.configure(path, aoi)
        return self.data_handler.handle(configuration)


class LocalFileReader(AbstractFileReader):
    def configure(self, path: str, aoi: AOI):
        return {"type": "local", "path": path, "aoi": aoi}


class MinioFileReader(AbstractFileReader):
    def __init__(self, minio_client, data_handler):
        super().__init__(data_handler)
        self.client = minio_client

    def configure(self, path: str, aoi: AOI):
        return {"type": "minio", "bucket": "your_bucket_name", "path": path, "aoi": aoi}
