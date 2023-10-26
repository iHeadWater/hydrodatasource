"""
Author: Wenyu Ouyang
Date: 2023-10-25 18:49:02
LastEditTime: 2023-10-26 09:24:10
LastEditors: Wenyu Ouyang
Description: Some configs for minio server
FilePath: \hydro_privatedata\hydroprivatedata\config.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""
import pathlib
import os
import boto3
from minio import Minio
import s3fs
from hydroutils import hydro_warning

MINIO_SERVER = "http://minio.waterism.com:9000"
LOCAL_DATA_PATH = None

minio_paras = {
    "endpoint_url": MINIO_SERVER,
    "access_key": "",
    "secret_key": "",
    "bucket_name": "test",
}

home_path = str(pathlib.Path.home())

if os.path.exists(os.path.join(home_path, ".wisminio")):
    for line in open(os.path.join(home_path, ".wisminio")):
        key = line.split("=")[0].strip()
        value = line.split("=")[1].strip()
        # print(key,value)
        if key == "endpoint_url":
            minio_paras["endpoint_url"] = value
        elif key == "access_key":
            minio_paras["access_key"] = value
        elif key == "secret_key":
            minio_paras["secret_key"] = value
        elif key == "bucket_path":
            minio_paras["bucket_name"] = value

if os.path.exists(os.path.join(home_path, ".hydrodataset")):
    for line in open(os.path.join(home_path, ".hydrodataset", "settings.txt")):
        LOCAL_DATA_PATH = line.strip()

if LOCAL_DATA_PATH is None:
    hydro_warning.no_directory(
        "LOCAL_DATA_PATH",
        "Please set LOCAL_DATA_PATH in ~/.hydrodataset, otherwise, you can't use the local data.",
    )

# Set up MinIO client
s3 = boto3.client(
    "s3",
    endpoint_url=MINIO_SERVER,
    aws_access_key_id=minio_paras["access_key"],
    aws_secret_access_key=minio_paras["secret_key"],
)
mc = Minio(
    MINIO_SERVER.replace("http://", ""),
    access_key=minio_paras["access_key"],
    secret_key=minio_paras["secret_key"],
    secure=False,
)
site_bucket = "sites"
site_object = "sites.csv"

fs = s3fs.S3FileSystem(
    client_kwargs={"endpoint_url": minio_paras["endpoint_url"]},
    key=minio_paras["access_key"],
    secret=minio_paras["secret_key"],
)

ro = {
    "client_kwargs": {"endpoint_url": minio_paras["endpoint_url"]},
    "key": minio_paras["access_key"],
    "secret": minio_paras["secret_key"],
}
