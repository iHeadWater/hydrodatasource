import xarray as xr
import pandas as pd
import numpy as np

import json
import tempfile
import os

from hydrodata.configs.config import (
    FS,
    GRID_INTERIM_BUCKET,
    MC,
    RO,
    DataConfig,
)
from hydrodata.processor.gpm import make_gpm_dataset
from hydrodata.processor.gfs import make_gfs_dataset


def merge_data(basin_id="1_02051500"):
    _data_config = DataConfig()
    data_config = _data_config.get_config()

    # 读取两个 NetCDF 文件
    if data_config["GPM_GFS_local_read"] is True:
        combined_data = xr.open_dataset(data_config["GPM_GFS_local_path"])

    elif data_config["GPM_GFS_merge"] is False:
        json_file_path = basin_id + "/gpm_gfs.json"
        # 从 MinIO 读取 JSON 文件
        with FS.open(f"{GRID_INTERIM_BUCKET}/{json_file_path}") as f:
            json_data = json.load(f)

        # 使用 xarray 和 kerchunk 读取数据
        combined_data = xr.open_dataset(
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

    else:
        gpm_data = make_gpm_dataset()
        gfs_data = make_gfs_dataset()

        # 指定时间段
        for time_num in data_config["time_periods"]:
            start_time = pd.to_datetime(time_num[0])
            end_time = pd.to_datetime(time_num[1])
        time_range = pd.date_range(start=start_time, end=end_time, freq="H")

        # 创建一个空的 xarray 数据集来存储结果
        combined_data = xr.Dataset()

        # 循环处理每个小时的数据
        for specified_time in time_range:
            m_hours = data_config["GPM_length"]  # 示例值，根据需要调整
            n_hours = data_config["GFS_length"]  # 示例值，根据需要调整

            gpm_data_filtered = gpm_data.sel(
                time=slice(specified_time - pd.Timedelta(hours=m_hours), specified_time)
            )
            gfs_data_filtered = gfs_data.sel(
                time=slice(
                    specified_time + pd.Timedelta(1),
                    specified_time + pd.Timedelta(hours=n_hours),
                )
            )

            gfs_data_interpolated = gfs_data_filtered.interp(
                lat=gpm_data.lat, lon=gpm_data.lon, method="linear"
            )
            combined_hourly_data = xr.concat(
                [gpm_data_filtered, gfs_data_interpolated], dim="time"
            )
            combined_hourly_data = combined_hourly_data.rename({"time": "step"})
            combined_hourly_data["step"] = np.arange(len(combined_hourly_data.step))
            time_now_hour = (
                specified_time
                - pd.Timedelta(hours=m_hours)
                + pd.Timedelta(hours=data_config["time_now"])
            )
            combined_hourly_data.coords["time_now"] = time_now_hour
            combined_hourly_data = combined_hourly_data.expand_dims("time_now")

            # 合并到结果数据集中
            combined_data = xr.merge(
                [combined_data, combined_hourly_data], combine_attrs="override"
            )

        if data_config["GPM_GFS_local_save"] is True:
            output_gpm_path = os.path.join(
                data_config["GPM_GFS_local_path"], "gpm_gfs.nc"
            )
            combined_data.to_netcdf(output_gpm_path)

        if data_config["GPM_GFS_upload"] is True:
            object_name = basin_id + "/gpm_gfs_test.nc"

            # 元数据
            time_periods_str = json.dumps(data_config["time_periods"])
            metadata = {
                "X-Amz-Meta-GPM_GFS_Time_Periods": time_periods_str,
            }

            with tempfile.NamedTemporaryFile() as tmp:
                # 将数据集保存到临时文件
                combined_data.to_netcdf(tmp.name)

                # 重置文件读取指针
                tmp.seek(0)

                # 上传到 MinIO
                MC.fput_object(
                    GRID_INTERIM_BUCKET, object_name, tmp.name, metadata=metadata
                )
    return combined_data
