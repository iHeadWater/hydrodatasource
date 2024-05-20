"""
Author: Wenyu Ouyang
Date: 2023-10-25 17:12:30
LastEditTime: 2024-03-28 08:33:13
LastEditors: Wenyu Ouyang
Description: Preprocess data source -- transform its format to tidy format
FilePath: \hydrodatasource\hydrodatasource\reader\stations.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import logging
import os
import pandas as pd

# from hydroutils import hydro_logger
from hydrodatasource.configs.config import (
    LOCAL_DATA_PATH,
    MC,
    STATION_BUCKET,
    STATION_OBJECT,
)
from hydrodatasource.reader.minio_api import minio_upload_csv


def huanren_preprocess():
    """
    Process data from huanren
    """
    # read local data
    try:
        huanren_df = pd.read_excel(
            os.path.join(LOCAL_DATA_PATH, "huanren", "桓仁多年逐日入库流量.xls"),
            sheet_name=1,
        )
        tidy_df2 = convert_to_tidy(huanren_df, "format2")
        local_save_path = os.path.join(LOCAL_DATA_PATH, "huanren", "tidy.csv")
        tidy_df2.to_csv(local_save_path, index=False)
        # boto3_upload_csv(s3, site_bucket, site_object, local_save_path)
        minio_upload_csv(MC, STATION_BUCKET, STATION_OBJECT, local_save_path)
    except Exception as e:
        preview = str(e)
        # hydro_logger.error(preview)
        logging.error(preview)


def biliu_stbprp_decode():
    biliu_stbprp_path = os.path.join(LOCAL_DATA_PATH, "biliu", "st_stbprp_b.xls")
    biliu_df = pd.read_excel(biliu_stbprp_path, sheet_name="st_stbprp_b")
    new_stids = []
    for stid in biliu_df["stid"]:
        new_stid = "2145" + str(stid).rjust(3, "0") + "X"
        new_stids.append(new_stid)
    biliu_df["stid"] = new_stids
    minio_upload_csv(
        MC, STATION_BUCKET, "test_stations/st_stbprp_b.xls", file_path=biliu_stbprp_path
    )
