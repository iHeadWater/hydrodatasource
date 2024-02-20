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
SETTING = {}
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