import xarray as xr
import pandas as pd
import tempfile
import os
import numpy as np
import json

from hydrodata.configs.config import (
    FS,
    GRID_INTERIM_BUCKET,
    LOCAL_DATA_PATH,
    MC,
    RO,
)
from hydrodata.processor.mask import generate_mask
from hydrodata.reader import minio


def make_gpm_dataset(
    basin_id="1_02051500",
    local_read=True,
    local_path=os.path.join(LOCAL_DATA_PATH, "gpm.nc"),
    minio_read=True,
):
    if local_read:
        latest_data = xr.open_dataset(local_path)

    elif minio_read:
        json_file_path = basin_id + "/gpm.json"
        # 从 MinIO 读取 JSON 文件
        with FS.open(f"{GRID_INTERIM_BUCKET}/{json_file_path}") as f:
            json_data = json.load(f)
        # 使用 xarray 和 kerchunk 读取数据
        latest_data = xr.open_dataset(
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
        latest_data = make1nc41basin(
            basin_id,
            local_path,
            mask_path=os.path.join(LOCAL_DATA_PATH, "mask"),
            shp_path=os.path.join(LOCAL_DATA_PATH, "shp"),
            dataset="camels",
            time_periods=[["2017-01-01T00:00:00", "2017-01-31T00:00:00"]],
            local_save=True,
            minio_upload=True,
        )

    return latest_data


def make1nc41basin(
    basin_id,
    local_path,
    mask_path=os.path.join(LOCAL_DATA_PATH, "mask"),
    shp_path=os.path.join(LOCAL_DATA_PATH, "shp"),
    dataset="camels",
    time_periods=[["2017-01-01T00:00:00", "2017-01-31T00:00:00"]],
    local_save=True,
    minio_upload=True,
):
    if dataset != "camels" and dataset != "wis":
        # 流域是国内还是国外，国外是camels，国内是wis
        raise ValueError("Invalid dataset")
    if os.path.isfile(mask_path):
        mask = xr.open_dataset(mask_path)
    else:
        shp_path = os.path.join(shp_path, basin_id, f"{basin_id}.shp")
        if not os.path.exists(mask_path):
            os.makedirs(mask_path)
        generate_mask(shp_path=shp_path, gpm_mask_folder_path=mask_path)
        mask_file_name = "mask-" + basin_id + "-gpm.nc"
        gpm_mask_file_path = os.path.join(mask_path, mask_file_name)
        mask = xr.open_dataset(gpm_mask_file_path)

    gpm_reader = minio.GPMReader()
    box = (
        mask.coords["lon"][0],
        mask.coords["lat"][0],
        mask.coords["lon"][-1],
        mask.coords["lat"][-1],
    )

    latest_data = xr.Dataset()
    for time_num in time_periods:
        start_time = np.datetime64(time_num[0])
        end_time = np.datetime64(time_num[1])
        data = gpm_reader.open_dataset(
            start_time=start_time,
            end_time=end_time,
            dataset=dataset,
            bbox=box,
            time_resolution="30m",
            time_chunks=24,
        )
        print(data)
        data = data.load()
        # data = data.to_dataset()
        # 转换时间维度至Pandas的DateTime格式并创建分组标签
        times = pd.to_datetime(data["time"].values)
        group_labels = times.floor("H")

        # 创建一个新的DataArray，其时间维度与原始数据集匹配
        group_labels_da = xr.DataArray(
            group_labels, coords={"time": data["time"]}, dims=["time"]
        )

        # 对数据进行分组并求和
        merge_gpm_data = data.groupby(group_labels_da).mean("time")

        # 将维度名称从group_labels_da重新命名为'time'
        merge_gpm_data = merge_gpm_data.rename({"group": "time"})
        merge_gpm_data.name = "tp"
        merge_gpm_data = merge_gpm_data.to_dataset()
        w_data = mask["w"]
        w_data_interpolated = w_data.interp(
            lat=merge_gpm_data.lat, lon=merge_gpm_data.lon, method="nearest"
        ).fillna(0)
        w_data_broadcasted = w_data_interpolated.broadcast_like(merge_gpm_data["tp"])
        merge_gpm_data = merge_gpm_data["tp"] * w_data_broadcasted
        if isinstance(latest_data, xr.DataArray):
            latest_data.name = "tp"
            latest_data = latest_data.to_dataset()
        if isinstance(merge_gpm_data, xr.DataArray):
            merge_gpm_data.name = "tp"
            merge_gpm_data = merge_gpm_data.to_dataset()
        is_empty = not latest_data.data_vars
        if is_empty:
            latest_data = merge_gpm_data
        else:
            latest_data = xr.concat([latest_data, merge_gpm_data], dim="time")

    if local_save:
        latest_data.to_netcdf(local_path)

    if minio_upload:
        object_name = basin_id + "/gpm_test.nc"

        # 元数据
        gpm_time_periods_str = json.dumps(time_periods)
        metadata = {
            "X-Amz-Meta-GPM_Time_Periods": gpm_time_periods_str,
        }

        with tempfile.NamedTemporaryFile() as tmp:
            # 将数据集保存到临时文件
            latest_data.to_netcdf(tmp.name)

            # 重置文件读取指针
            tmp.seek(0)

            # 上传到 MinIO
            MC.fput_object(
                GRID_INTERIM_BUCKET, object_name, tmp.name, metadata=metadata
            )

    return latest_data
