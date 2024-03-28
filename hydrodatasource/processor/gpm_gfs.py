from datetime import timedelta
import datetime
import xarray as xr
import pandas as pd
import tempfile
import os
import numpy as np
import json

from hydrodatasource.configs.config import (
    FS,
    GRID_INTERIM_BUCKET,
    LOCAL_DATA_PATH,
    MC,
    RO,
)
from hydrodatasource.utils.utils import generate_time_intervals
from hydrodatasource.processor.mask import gen_single_mask
from hydrodatasource.reader import minio


def process_data(initial_date, initial_time, mask, dataset):
    # Convert the initial date and time to a datetime object
    current_datetime = datetime.datetime.strptime(
        f"{initial_date} {initial_time}", "%Y-%m-%d %H"
    )
    initial_datetime = datetime.datetime.strptime(
        f"{initial_date} {initial_time}", "%Y-%m-%d %H"
    )
    gfs_reader = minio.GFSReader()
    box = (
        mask.coords["lon"][0] - 0.2,
        mask.coords["lat"][0] - 0.2,
        mask.coords["lon"][-1] + 0.2,
        mask.coords["lat"][-1] + 0.2,
    )

    while True:
        # Your existing code to process the data
        data = gfs_reader.open_dataset(
            creation_date=np.datetime64(current_datetime.strftime("%Y-%m-%d")),
            creation_time=current_datetime.strftime("%H"),
            bbox=box,
            dataset=dataset,
            time_chunks=24,
        )
        data = data.load()
        data = data.to_dataset()
        data = data.transpose("time", "valid_time", "lon", "lat")

        # Adjust the slicing based on the current datetime
        total_hours_passed = int(
            (initial_datetime - current_datetime).total_seconds() / 3600
        )

        # Adjust the slicing based on the total hours passed
        start_slice = total_hours_passed + 1
        end_slice = start_slice + 6
        # Minio上存储的gfs数据都是该时刻未来120小时后的数据，因此这里设置了一个报错
        if start_slice > 120:
            raise Exception("This datetime does not have enough data to merge")
        data = data.isel(time=0, valid_time=slice(start_slice, end_slice))

        # Check for negative values
        data_det = data["total_precipitation_surface"].values
        has_negative = data_det < 0
        if has_negative.any():
            # If negative, roll back time by 6 hours and repeat
            current_datetime -= timedelta(hours=6)
        else:
            # No negative values found, return the data
            return data


def make_gpm_dataset(
    time_periods,
    dataset,
    mask,
):
    gpm_reader = minio.GPMReader()
    latest_data = xr.Dataset()
    box = (
        mask.coords["lon"][0],
        mask.coords["lat"][0],
        mask.coords["lon"][-1],
        mask.coords["lat"][-1],
    )
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
        data = data.load()
        # data = data.to_dataset()
        # 转换时间维度至Pandas的DateTime格式并创建分组标签
        times = pd.to_datetime(data["time"].values)
        group_labels = times.floor("h")

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
    return latest_data


def make_gfs_dataset(
    time_periods,
    dataset,
    mask,
):
    final_latest_data = xr.Dataset()
    for time_num in time_periods:
        start_time = datetime.datetime.strptime(time_num[0], "%Y-%m-%dT%H:%M:%S")
        end_time = datetime.datetime.strptime(time_num[1], "%Y-%m-%dT%H:%M:%S")
        gfs_time_list = generate_time_intervals(start_time, end_time)
        w_data = mask["w"]
        latest_data = xr.Dataset()

        for date, time in gfs_time_list:
            data = process_data(date, time, mask, dataset)
            if isinstance(data, xr.DataArray):
                data = data.to_dataset()
            if "total_precipitation_surface" in data.data_vars:
                data = data.rename({"total_precipitation_surface": "tp"})
            data = data.drop_vars("time")
            # print(data)
            data = data.rename({"valid_time": "time"})
            w_data_interpolated = w_data.interp(
                lat=data.lat, lon=data.lon, method="nearest"
            ).fillna(0)

            # 将 w 数据广播到与当前数据集相同的时间维度上
            w_data_broadcasted = w_data_interpolated.broadcast_like(data["tp"])
            data["tp"] = data["tp"] * w_data_broadcasted
            if isinstance(latest_data, xr.DataArray):
                latest_data.name = "tp"
                latest_data = latest_data.to_dataset()
            if isinstance(data, xr.DataArray):
                data.name = "tp"
                data = data.to_dataset()
            is_empty = not latest_data.data_vars
            latest_data = (
                data if is_empty else xr.concat([latest_data, data], dim="time")
            )
        if isinstance(latest_data, xr.DataArray):
            latest_data.name = "tp"
            latest_data = latest_data.to_dataset()
        if isinstance(final_latest_data, xr.DataArray):
            final_latest_data.name = "tp"
            final_latest_data = final_latest_data.to_dataset()
        is_final_empty = not final_latest_data.data_vars
        if is_final_empty:
            final_latest_data = latest_data
        else:
            final_latest_data = xr.concat([final_latest_data, latest_data], dim="time")
    return final_latest_data


def make_merge_dataset(
    gpm_data, gfs_data, time_periods, gpm_length, gfs_length, time_now_length
):
    # 指定时间段
    for time_num in time_periods:
        start_time = pd.to_datetime(time_num[0])
        end_time = pd.to_datetime(time_num[1])
    time_range = pd.date_range(start=start_time, end=end_time, freq="H")

    # 创建一个空的 xarray 数据集来存储结果
    combined_data = xr.Dataset()

    # 循环处理每个小时的数据
    for specified_time in time_range:
        m_hours = gpm_length
        n_hours = gfs_length

        # 选取specified_time前指定时间段的时间的gpm数据
        gpm_data_filtered = gpm_data.sel(
            time=slice(specified_time - pd.Timedelta(hours=m_hours), specified_time)
        )
        # 选取specified_time后指定时间段的时间的gfs数据
        gfs_data_filtered = gfs_data.sel(
            time=slice(
                specified_time + pd.Timedelta(1),  # gfs数据要在
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
            + pd.Timedelta(hours=time_now_length)
        )
        combined_hourly_data.coords["time_now"] = time_now_hour
        combined_hourly_data = combined_hourly_data.expand_dims("time_now")

        # 合并到结果数据集中
        combined_data = xr.merge(
            [combined_data, combined_hourly_data], combine_attrs="override"
        )
    return combined_data


def make1nc41basin(
    basin_id="1_02051500",
    dataname="gpm",
    local_path=LOCAL_DATA_PATH,
    mask_path=os.path.join(LOCAL_DATA_PATH, "dataset-origin", "mask"),
    shp_path=os.path.join(LOCAL_DATA_PATH, "dataset-origin", "shp"),
    dataset="camels",
    time_periods=[["2017-01-01T00:00:00", "2017-01-31T00:00:00"]],
    local_save=True,
    minio_upload=False,
    gpm_length=None,
    gfs_length=None,
    time_now_length=None,
    gpm_path=None,
    gfs_path=None,
):
    if dataset not in ["camels", "wis"]:
        # 流域是国内还是国外，国外是camels，国内是wis
        raise ValueError("Invalid dataset")

    mask = gen_single_mask(basin_id, shp_path, dataname, mask_path)

    # 如果是合并数据，需要额外处理
    if dataname == "merge":
        mask = gen_single_mask(basin_id, shp_path, "gpm", mask_path)
        gpm_data = (
            make_gpm_dataset(time_periods, dataset, mask)
            if gpm_path is None
            else xr.open_dataset(gpm_path)
        )
        mask = gen_single_mask(basin_id, shp_path, "gfs", mask_path)
        gfs_data = (
            make_gfs_dataset(time_periods, dataset, mask)
            if gfs_path is None
            else xr.open_dataset(gfs_path)
        )

    data_functions = {
        "gpm": lambda: make_gpm_dataset(time_periods, dataset, mask),
        "gfs": lambda: make_gfs_dataset(time_periods, dataset, mask),
        "merge": lambda: make_merge_dataset(
            gpm_data, gfs_data, time_periods, gpm_length, gfs_length, time_now_length
        ),
    }

    # 检查数据名称是否有效
    if dataname not in data_functions:
        raise NotImplementedError(
            f"This type of data ({dataname}) is not available for now, please try gpm, gfs, or merge"
        )

    # 根据数据名称调用相应的函数
    latest_data = data_functions[dataname]()

    if local_save:
        local_save_path = os.path.join(local_path, "datasets-interim", basin_id)
        if not os.path.exists(local_save_path):
            os.makedirs(local_save_path)
        local_file_path = os.path.join(
            local_path, "datasets-interim", basin_id, dataname
        )
        local_file_name = local_file_path + ".nc"
        latest_data.to_netcdf(local_file_name)

    if minio_upload:
        object_name = basin_id + "/" + dataname + ".nc"

        # 元数据
        time_periods_str = json.dumps(time_periods)
        metadata = {
            "X-Amz-Meta-Time_Periods": time_periods_str,
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
