"""
Author: Wenyu Ouyang
Date: 2024-03-24 08:48:57
LastEditTime: 2025-03-12 10:02:27
LastEditors: Wenyu Ouyang
Description: read meta data from minio
FilePath: /hydrodatasource/hydrodatasource/reader/metadata.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import subprocess
import json
import time
from hydrodatasource.configs.config import FS


def list_files_s3fs(bucket_name):
    start_time = time.time()  # record start time

    files = []
    for path in FS.glob(f"{bucket_name}/**"):
        if FS.isfile(path):
            info = FS.info(path)
            files.append(
                {
                    "name": info["Key"].split("/", 1)[1],
                    "size": info["Size"],
                    "last_modified": info["LastModified"],
                }
            )

    elapsed = time.time() - start_time  # calculate elapsed time
    print(
        f"[s3fs] Processing completed | Number of files: {len(files)} | Time elapsed: {elapsed:.2f}s"
    )
    return files


def list_files_rclone(bucket_name):
    start_time = time.time()
    cmd = [
        "rclone",
        "lsjson",
        f"minio/{bucket_name}",
        "--fast-list",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, check=True)

    files = json.loads(result.stdout)
    filtered_files = [
        {"name": f["Path"], "size": f["Size"], "last_modified": f["ModTime"]}
        for f in files
        if not f["IsDir"]
    ]

    elapsed = time.time() - start_time
    print(
        f"[rclone] Processing completed | Number of files: {len(filtered_files)} | Time elapsed: {elapsed:.2f}s"
    )
    return filtered_files


if __name__ == "__main__":
    s3fs_files = list_files_s3fs("yuque")
    rclone_files = list_files_rclone("yuque")
