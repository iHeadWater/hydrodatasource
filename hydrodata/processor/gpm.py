from hydro_opendata.processor.mask import generate_mask
import xarray as xr
from hydro_opendata.reader import minio
import pandas as pd
from hydro_opendata.configs.config import DataConfig
from minio import Minio
import tempfile
import os
import numpy as np
import json
import s3fs
from kerchunk.hdf import SingleHdf5ToZarr


def make_gpm_dataset():
    _data_config = DataConfig()
    data_config = _data_config.get_config()
    
    if data_config["GPM_local_read"] is True:
        latest_data = xr.open_dataset(data_config["GPM_local_path"])
        
    elif data_config["GPM_merge"] is False:
        minio_url = 'http://minio.waterism.com:9000'
        access_key = data_config["MINIO_access_key"]
        secret_key = data_config["MINIO_secret_key"]
        bucket_name = 'grids-interim'
        json_file_path = data_config["basin_id"] + '/gpm.json'
        fs = s3fs.S3FileSystem(client_kwargs={'endpoint_url': minio_url}, key=access_key, secret=secret_key, use_ssl=False)

        # 从 MinIO 读取 JSON 文件
        with fs.open(f'{bucket_name}/{json_file_path}') as f:
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
                    "remote_options": {'client_kwargs': {'endpoint_url': minio_url},
                                    'key': access_key, 'secret': secret_key, 'use_ssl': False}
                }
            }
        )
        
    else:     
        gpm_mask_path = data_config["GPM_mask_path"]
        if data_config["GPM_mask"] is True:
            mask = xr.open_dataset(gpm_mask_path) 
        else:
            shp_path = data_config["shp_path"]
            if not os.path.exists(gpm_mask_path):
                os.makedirs(gpm_mask_path)
            generate_mask(shp_path = shp_path, gpm_mask_folder_path=gpm_mask_path)
            mask_file_name = "mask-" + data_config["basin_id"] + "-gpm.nc"
            gpm_mask_file_path = os.path.join(gpm_mask_path, mask_file_name)
            mask = xr.open_dataset(gpm_mask_file_path)

        gpm_reader = minio.GPMReader()
        box = (
                mask.coords["lon"][0],
                mask.coords["lat"][0],
                mask.coords["lon"][-1],
                mask.coords["lat"][-1],
        )
        
        latest_data = xr.Dataset()
        for time_num in data_config["GPM_time_periods"]:
            start_time = np.datetime64(time_num[0])
            end_time =  np.datetime64(time_num[1])
            dataset = data_config["dataset"]
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
            times = pd.to_datetime(data['time'].values)
            group_labels = times.floor('H')

            # 创建一个新的DataArray，其时间维度与原始数据集匹配
            group_labels_da = xr.DataArray(group_labels, coords={'time': data['time']}, dims=['time'])

            # 对数据进行分组并求和
            merge_gpm_data = data.groupby(group_labels_da).mean('time')

            # 将维度名称从group_labels_da重新命名为'time'
            merge_gpm_data = merge_gpm_data.rename({'group': 'time'})
            merge_gpm_data.name = 'tp'
            merge_gpm_data = merge_gpm_data.to_dataset()
            w_data = mask["w"]
            w_data_interpolated = w_data.interp(
                        lat=merge_gpm_data.lat, lon=merge_gpm_data.lon, method="nearest"
                    ).fillna(0)
            w_data_broadcasted = w_data_interpolated.broadcast_like(
                        merge_gpm_data["tp"]
                    )
            merge_gpm_data = merge_gpm_data["tp"] * w_data_broadcasted
            if isinstance(latest_data, xr.DataArray):
                latest_data.name = 'tp'
                latest_data = latest_data.to_dataset()
            if isinstance(merge_gpm_data, xr.DataArray):
                merge_gpm_data.name = 'tp'
                merge_gpm_data = merge_gpm_data.to_dataset()
            is_empty = not latest_data.data_vars
            if is_empty:
                latest_data = merge_gpm_data
            else:
                latest_data = xr.concat([latest_data, merge_gpm_data], dim="time")
        
        if data_config["GPM_local_save"] is True:
            output_gpm_path = os.path.join(data_config["GPM_local_path"], "gpm.nc")
            latest_data.to_netcdf(output_gpm_path)
        
        if data_config["GPM_upload"] is True:
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
            object_name = data_config["basin_id"] + '/gpm_test.nc'
            
            # 元数据
            gpm_time_periods_str = json.dumps(data_config["GPM_time_periods"])
            metadata = {
                'X-Amz-Meta-GPM_Time_Periods': gpm_time_periods_str,
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
            
    return latest_data