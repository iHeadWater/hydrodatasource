"""
Author: Wenyu Ouyang
Date: 2023-10-25 17:12:30
LastEditTime: 2023-10-26 09:19:22
LastEditors: Wenyu Ouyang
Description: Preprocess data source, maybe we need to process manually
FilePath: \hydro_privatedata\hydroprivatedata\preprocess.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""
import logging
import os
import pandas as pd
# from hydroutils import hydro_logger
from hydroprivatedata.config import (
    LOCAL_DATA_PATH,
    s3,
    mc,
    site_bucket,
    site_object,
)
from hydroprivatedata.minio_api import boto3_upload_csv, minio_upload_csv
from hydroprivatedata.source_data_dict import convert_to_tidy


def huanren_preprocess():
    """
    Process data from huanren
    """
    # read local data
    try:
        huanren_df = pd.read_excel(
            os.path.join(LOCAL_DATA_PATH, "huanren", "桓仁多年逐日入库流量.xls"), sheet_name=1
        )
        tidy_df2 = convert_to_tidy(huanren_df, "format2")
        local_save_path = os.path.join(LOCAL_DATA_PATH, "huanren", "tidy.csv")
        tidy_df2.to_csv(local_save_path, index=False)
        # boto3_upload_csv(s3, site_bucket, site_object, local_save_path)
        minio_upload_csv(mc, site_bucket, site_object, local_save_path)
    except Exception as e:
        preview = str(e)
        # hydro_logger.error(preview)
        logging.error(preview)
