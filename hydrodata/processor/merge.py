import xarray as xr
import pandas as pd
import numpy as np
from hydro_opendata.configs.config import DataConfig
from hydro_opendata.processor.gpm import make_gpm_dataset
from hydro_opendata.processor.gfs import make_gfs_dataset
import json
import s3fs
from kerchunk.hdf import SingleHdf5ToZarr
import tempfile
from minio import Minio
import os


def merge_data():
    _data_config = DataConfig()
    data_config = _data_config.get_config()

    # 读取两个 NetCDF 文件
    if data_config["GPM_GFS_local_read"] is True:
        combined_data = xr.open_dataset(data_config["GPM_GFS_local_path"])
    
    elif data_config["GPM_GFS_merge"] is False:
        minio_url = 'http://minio.waterism.com:9000'
        access_key = data_config["MINIO_access_key"]
        secret_key = data_config["MINIO_secret_key"]
        bucket_name = 'grids-interim'
        json_file_path = data_config["basin_id"] + '/gpm_gfs.json'
        fs = s3fs.S3FileSystem(client_kwargs={'endpoint_url': minio_url}, key=access_key, secret=secret_key, use_ssl=False)

        # 从 MinIO 读取 JSON 文件
        with fs.open(f'{bucket_name}/{json_file_path}') as f:
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
                    "remote_options": {'client_kwargs': {'endpoint_url': minio_url},
                                    'key': access_key, 'secret': secret_key, 'use_ssl': False}
                }
            }
        )
    
    else:
        gpm_data = make_gpm_dataset()
        gfs_data = make_gfs_dataset()

        # 指定时间段
        for time_num in data_config["time_periods"]:
            start_time = pd.to_datetime(time_num[0])
            end_time = pd.to_datetime(time_num[1])
        time_range = pd.date_range(start=start_time, end=end_time, freq='H')

        # 创建一个空的 xarray 数据集来存储结果
        combined_data = xr.Dataset()

        # 循环处理每个小时的数据
        for specified_time in time_range:
            m_hours = data_config["GPM_length"]  # 示例值，根据需要调整
            n_hours = data_config["GFS_length"]  # 示例值，根据需要调整
            
            gpm_data_filtered = gpm_data.sel(time=slice(specified_time - pd.Timedelta(hours=m_hours), specified_time))
            gfs_data_filtered = gfs_data.sel(time=slice(specified_time + pd.Timedelta(1), specified_time + pd.Timedelta(hours=n_hours)))
            
            gfs_data_interpolated = gfs_data_filtered.interp(lat=gpm_data.lat, lon=gpm_data.lon, method='linear')
            combined_hourly_data = xr.concat([gpm_data_filtered, gfs_data_interpolated], dim='time')
            combined_hourly_data = combined_hourly_data.rename({'time': 'step'})
            combined_hourly_data['step'] = np.arange(len(combined_hourly_data.step))
            time_now_hour = specified_time - pd.Timedelta(hours=m_hours) + pd.Timedelta(hours=data_config["time_now"])
            combined_hourly_data.coords['time_now'] = time_now_hour
            combined_hourly_data = combined_hourly_data.expand_dims('time_now')
            
            # 合并到结果数据集中
            combined_data = xr.merge([combined_data, combined_hourly_data], combine_attrs="override")

        if data_config["GPM_GFS_local_save"] is True:
            output_gpm_path = os.path.join(data_config["GPM_GFS_local_path"], "gpm_gfs.nc")
            combined_data.to_netcdf(output_gpm_path)
        
        if data_config["GPM_GFS_upload"] is True:
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
            object_name = data_config["basin_id"] + '/gpm_gfs_test.nc'
            
            # 元数据
            time_periods_str = json.dumps(data_config["time_periods"])
            metadata = {
                'X-Amz-Meta-GPM_GFS_Time_Periods': time_periods_str,
            }
            
            with tempfile.NamedTemporaryFile() as tmp:
                # 将数据集保存到临时文件
                combined_data.to_netcdf(tmp.name)

                # 重置文件读取指针
                tmp.seek(0)

                # 上传到 MinIO
                client.fput_object(
                    bucket_name, object_name, tmp.name,
                    metadata = metadata
                )
    return combined_data
