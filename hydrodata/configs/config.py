"""
Author: Jianfeng Zhu
Date: 2023-10-25 18:49:02
LastEditTime: 2024-02-15 21:08:19
LastEditors: Wenyu Ouyang
Description: Some configs for minio server
FilePath: \hydrodata\hydrodata\configs\config.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import os
from pathlib import Path
import boto3
import s3fs
import yaml
from minio import Minio


def read_setting(setting_path):
    if not os.path.exists(setting_path):
        raise FileNotFoundError(f"Configuration file not found: {setting_path}")

    with open(setting_path, "r") as file:
        setting = yaml.safe_load(file)

    example_setting = (
        "minio:\n"
        "  server_url: 'http://minio.waterism.com:9090' # Update with your URL\n"
        "  client_endpoint: 'http://minio.waterism.com:9000' # Update with your URL\n"
        "  access_key: 'your minio access key'\n"
        "  secret: 'your minio secret'\n\n"
        "local_data_path:\n"
        "  root: 'D:\\data\\waterism' # Update with your root data directory\n"
        "  datasets-origin: 'D:\\data\\waterism\\datasets-origin'\n"
        "  datasets-interim: 'D:\\data\\waterism\\datasets-interim'"
    )

    if setting is None:
        raise ValueError(
            f"Configuration file is empty or has invalid format.\n\nExample configuration:\n{example_setting}"
        )

    # Define the expected structure
    expected_structure = {
        "minio": ["server_url", "client_endpoint", "access_key", "secret"],
        "local_data_path": ["root", "datasets-origin", "datasets-interim"],
    }

    # Validate the structure
    try:
        for key, subkeys in expected_structure.items():
            if key not in setting:
                raise KeyError(f"Missing required key in config: {key}")

            if isinstance(subkeys, list):
                for subkey in subkeys:
                    if subkey not in setting[key]:
                        raise KeyError(f"Missing required subkey '{subkey}' in '{key}'")
    except KeyError as e:
        raise ValueError(
            f"Incorrect configuration format: {e}\n\nExample configuration:\n{example_setting}"
        ) from e

    return setting


SETTING_FILE = os.path.join(Path.home(), "hydro_setting.yml")
try:
    SETTING = read_setting(SETTING_FILE)
except ValueError as e:
    print(e)
except Exception as e:
    print(f"Unexpected error: {e}")

LOCAL_DATA_PATH = SETTING["local_data_path"]["root"]

MINIO_PARAM = {
    "endpoint_url": SETTING["minio"]["server_url"],
    "access_key": SETTING["minio"]["access_key"],
    "secret_key": SETTING["minio"]["secret"],
    "bucket_name": "test",
}

FS = s3fs.S3FileSystem(
    client_kwargs={"endpoint_url": MINIO_PARAM["endpoint_url"]},
    key=MINIO_PARAM["access_key"],
    secret=MINIO_PARAM["secret_key"],
    use_ssl=False,
)

# remote_options parameters for xr open_dataset from minio
RO = {
    "client_kwargs": {"endpoint_url": MINIO_PARAM["endpoint_url"]},
    "key": MINIO_PARAM["access_key"],
    "secret": MINIO_PARAM["secret_key"],
    "use_ssl": False,
}


# Set up MinIO client
S3 = boto3.client(
    "s3",
    endpoint_url=SETTING["minio"]["server_url"],
    aws_access_key_id=MINIO_PARAM["access_key"],
    aws_secret_access_key=MINIO_PARAM["secret_key"],
)
MC = Minio(
    SETTING["minio"]["server_url"].replace("http://", ""),
    access_key=MINIO_PARAM["access_key"],
    secret_key=MINIO_PARAM["secret_key"],
    secure=False,  # True if using HTTPS
)
STATION_BUCKET = "stations"
STATION_OBJECT = "sites.csv"

GRID_INTERIM_BUCKET = "grids-interim"


class DataConfig:
    def __init__(self):
        self.config = {
            # 是否从hydrodata读取GPM_GFS文件
            "GPM_GFS_local_read": False,
            "GPM_GFS_merge": True,
            # 如果从本地读，需要给定GPM_GFS的路径,upload和local代表是否上传到Minio和是否下载到本地
            "GPM_GFS_local_path": "/home/xushuolong1/hydrodata/data",
            "GPM_GFS_upload": False,
            "GPM_GFS_local_save": True,
            # 是否需要从源数据合并新数据
            "GFS_merge": True,
            # 如果从本地读，需要给定GPM或者GFS的路径
            "GFS_local_path": "/home/xushuolong1/hydrodata/data/gfs.nc",
            # 数据类型，目前只有降水
            "data_type": "tp",
            # 拼接后的gpm_gfs的step中，GPM数据占据前多少个step
            "GPM_length": 169,
            # 拼接后的gpm_gfs的step中，GFS数据占据后多少个step
            "GFS_length": 23,
            # 拼接后的gpm_gfs的time_now，对应step中的第多少个值
            "time_now": 168,
            # 拼接后的gpm_gfs的time_now的时间范围
            "time_periods": [
                ["2017-01-10T00:00:00", "2017-01-11T00:00:00"],
                ["2017-01-15T00:00:00", "2017-01-20T00:00:00"],
            ],
            # 是否从hydrodata重新读取并生成gpm.nc，是否上传，是否保存在本地
            # GFS基本同理，额外会有两个参数
            # enlarge参数，是因为GPM和GFS拼接的时候，需要GFS比GPM大一圈才能插值，不然会出现-
            # new参数，这个是为了模拟实际预报，因为实际预报的时候，GFS的数据不会是每一个时刻最新的数据
            # 但考虑到有时候算其他数据不需要大一圈，因此这里写了个参数
            "GFS_enlarge": True,
            "GFS_new": True,
            "GFS_upload": False,
            "GFS_local_save": True,
            # 如果重新读取并生成gpm.nc，且没有mask.nc，那么需要提供GPM_mask，GFS同理
            "GFS_mask": False,
            "GFS_mask_path": "/home/xushuolong1/hydrodata/data/mask",
            # 如果需要生成mask.nc，那么需要提供流域的shp文件
            # 如果生成GPM.nc，那么需要提供时间范围，理论上最好是连续的，以防特殊需求或者特殊情况，这里还是写成了分段的，GFS同理
            "GFS_time_periods": [
                ["2017-01-01T00:00:00", "2017-01-31T00:00:00"],
            ],
        }

    def get_config(self):
        return self.config
