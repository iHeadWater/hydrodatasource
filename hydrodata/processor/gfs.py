from datetime import timedelta, datetime
from hydro_opendata.reader import minio
from hydro_opendata.processor.mask import generate_mask
from hydro_opendata.configs.config import DataConfig
import numpy as np
import xarray as xr
import os
from minio import Minio
import tempfile
import json
import s3fs
from kerchunk.hdf import SingleHdf5ToZarr

# 生成开始时间到结束时间的一个用于GFS数据读取的列表    
def generate_time_intervals(start_date, end_date):
    # Convert input strings to datetime objects
    start = start_date
    end = end_date
    # start = datetime.strptime(start_date, "%Y-%m-%d")
    # end = datetime.strptime(end_date, "%Y-%m-%d")

    # Initialize an empty list to store the intervals
    intervals = []

    # Loop over days
    while start <= end:
        # Loop over the four time intervals in a day
        for hour in ["00", "06", "12", "18"]:
            intervals.append([start.strftime("%Y-%m-%d"), hour])
        
        # Move to the next day
        start += timedelta(days=1)

    return intervals

# 检索数据是否为空，返回不为空的数据
def process_data(initial_date, initial_time):
    # Convert the initial date and time to a datetime object
    current_datetime = datetime.strptime(f"{initial_date} {initial_time}", "%Y-%m-%d %H")
    initial_datetime = datetime.strptime(f"{initial_date} {initial_time}", "%Y-%m-%d %H")
    gfs_reader = minio.GFSReader()
    gfs_mask_path = "/home/xushuolong1/biliuhe_test/gfs_mask/mask-1-gfs.nc"
    mask = xr.open_dataset(gfs_mask_path)
    box = (
        mask.coords["lon"][0] - 0.2,
        mask.coords["lat"][0] - 0.2,
        mask.coords["lon"][-1] + 0.2,
        mask.coords["lat"][-1] + 0.2,
    )

    while True:
        # Your existing code to process the data
        data = gfs_reader.open_dataset(
            # data_variable="tp",
            creation_date=np.datetime64(current_datetime.strftime("%Y-%m-%d")),
            creation_time=current_datetime.strftime("%H"),
            bbox=box,
            dataset="wis",
            time_chunks=24,
        )
        data = data.load()
        data = data.to_dataset()
        data = data.transpose("time", "valid_time", "lon", "lat")

        # Adjust the slicing based on the current datetime
        total_hours_passed = int((initial_datetime - current_datetime).total_seconds() / 3600)

        # Adjust the slicing based on the total hours passed
        start_slice = total_hours_passed + 1
        end_slice = start_slice + 6
        if start_slice > 120:
            print(np.datetime64(current_datetime.strftime("%Y-%m-%d")))
            raise Exception('This datetime does not have enough data to merge')
        data = data.isel(time=0, valid_time=slice(start_slice, end_slice))
        # print(data['total_precipitation_surface'].values)

        # Check for negative values
        data_det = data['total_precipitation_surface'].values
        has_negative = data_det < 0
        any_negative = has_negative.any()
        # print(any_negative)

        if not any_negative:
            # No negative values found, return the data
            return data

        # If negative, roll back time by 6 hours and repeat
        current_datetime -= timedelta(hours=6)


def make_gfs_dataset():
    _data_config = DataConfig()
    data_config = _data_config.get_config()
    if data_config["GFS_local_read"] is True:
        final_latest_data = xr.open_dataset(data_config["GFS_local_path"])
        
    elif data_config["GFS_merge"] is False:
        minio_url = 'http://minio.waterism.com:9000'
        access_key = data_config["MINIO_access_key"]
        secret_key = data_config["MINIO_secret_key"]
        bucket_name = 'grids-interim'
        json_file_path = data_config["basin_id"] + '/gfs.json'
        fs = s3fs.S3FileSystem(client_kwargs={'endpoint_url': minio_url}, key=access_key, secret=secret_key, use_ssl=False)

        # 从 MinIO 读取 JSON 文件
        with fs.open(f'{bucket_name}/{json_file_path}') as f:
            json_data = json.load(f)

        # 使用 xarray 和 kerchunk 读取数据
        final_latest_data = xr.open_dataset(
            "reference://",
            engine="zarr",
            backend_kwargs={
                "consolidated": False,
                "storage_options": {
                    "fo": json_data,
                    "remote_protocol": "s3",
                    "remote_options": {'client_kwargs': {'endpoint_url': minio_url},
                                    'key': access_key, 'secret': secret_key, 'use_ssl': False}
                }
            }
        )  
        
    else:  
        gfs_mask_path = data_config["GFS_mask_path"]
        if data_config["GFS_mask"] is True:
            mask = xr.open_dataset(gfs_mask_path) 
        else:
            shp_path = data_config["shp_path"]
            if not os.path.exists(gfs_mask_path):
                os.makedirs(gfs_mask_path)
            generate_mask(shp_path = shp_path, gfs_mask_folder_path=gfs_mask_path)
            mask_file_name = "mask-" + data_config["basin_id"] + "-gfs.nc"
            gfs_mask_file_path = os.path.join(gfs_mask_path, mask_file_name)
            mask = xr.open_dataset(gfs_mask_file_path)
        
        final_latest_data = xr.Dataset()
        for time_num in data_config["GFS_time_periods"]:
            start_time = datetime.strptime(time_num[0], "%Y-%m-%dT%H:%M:%S")
            end_time = datetime.strptime(time_num[1], "%Y-%m-%dT%H:%M:%S")    
            gfs_time_list = generate_time_intervals(start_time, end_time)
            w_data = mask["w"]
            latest_data = xr.Dataset()

            for date, time in gfs_time_list:
                data = process_data(date, time)
                if isinstance(data, xr.DataArray):
                    data = data.to_dataset()
                if "total_precipitation_surface" in data.data_vars:
                    data  = data.rename({'total_precipitation_surface': 'tp'})
                data =data.drop_vars("time")
                # print(data)
                data = data.rename({"valid_time": "time"})
                w_data_interpolated = w_data.interp(
                        lat=data.lat, lon=data.lon, method="nearest"
                    ).fillna(0)

                # 将 w 数据广播到与当前数据集相同的时间维度上
                w_data_broadcasted = w_data_interpolated.broadcast_like(
                    data["tp"]
                )
                data['tp'] = data["tp"] * w_data_broadcasted
                if isinstance(latest_data, xr.DataArray):
                    latest_data.name = 'tp'
                    latest_data = latest_data.to_dataset()
                if isinstance(data, xr.DataArray):
                    data.name = 'tp'
                    data = data.to_dataset()
                is_empty = not latest_data.data_vars
                if is_empty:
                    latest_data = data
                else:
                    latest_data = xr.concat([latest_data, data], dim="time")
            if isinstance(latest_data, xr.DataArray):
                latest_data.name = 'tp'
                latest_data = latest_data.to_dataset()
            if isinstance(final_latest_data, xr.DataArray):
                final_latest_data.name = 'tp'
                final_latest_data = final_latest_data.to_dataset()
            is_final_empty = not final_latest_data.data_vars   
            if is_final_empty:
                final_latest_data = latest_data
            else:
                final_latest_data = xr.concat([final_latest_data, latest_data], dim="time")
        
        if data_config["GFS_local_save"] is True:
            output_gfs_path = os.path.join(data_config["GFS_local_path"], "gfs.nc")
            latest_data.to_netcdf(output_gfs_path)
        
        if data_config["GFS_upload"] is True:
            minio_url = 'minio.waterism.com:9000'
            access_key = data_config["MINIO_access_key"]
            secret_key = data_config["MINIO_secret_key"]
            
            client = Minio(
                endpoint= minio_url,  # 替换为您的 MinIO 服务器地址
                access_key=access_key,  # 替换为您的访问密钥
                secret_key=secret_key,  # 替换为您的秘密密钥
                secure=False  # 设置为 True 如果使用 HTTPS
            )
            bucket_name = "grids-interim"
            object_name = data_config["basin_id"] + '/gfs_test.nc'
            
            # 元数据
            gfs_time_periods_str = json.dumps(data_config["GFS_time_periods"])
            metadata = {
                'X-Amz-Meta-GFS_Time_Periods': gfs_time_periods_str,
            }
            
            with tempfile.NamedTemporaryFile() as tmp:
                # 将数据集保存到临时文件
                latest_data.to_netcdf(tmp.name)

                # 重置文件读取指针
                tmp.seek(0)

                # 上传到 MinIO
                client.fput_object(
                    bucket_name, object_name, tmp.name,
                    metadata = metadata
                )
            
    return final_latest_data