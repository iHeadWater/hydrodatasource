"""
Author: Wenyu Ouyang
Date: 2025-01-06 20:34:34
LastEditTime: 2025-01-07 09:26:39
LastEditors: Wenyu Ouyang
Description: script for chinese streamflow preprocessing
FilePath: \hydrodatasource\scripts\chinese_streamflow_preprocessing.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import os
import sys
from pathlib import Path


sys.path.append(os.path.dirname(Path(os.path.abspath(__file__)).parent))
from const4scripts import RESULT_DIR, DATASET_DIR
from hydrodatasource.cleaner.streamflow_cleaner import StreamflowBacktrack

# 测试径流数据反推处理功能
original_reservoir_data_dir = os.path.join(DATASET_DIR, "数据库原始流量")
tmp_dir = os.path.join(RESULT_DIR, "反推流量")
cleaner = StreamflowBacktrack(
    data_folder=original_reservoir_data_dir,
    output_folder=tmp_dir,
)
cleaner.process_backtrack()


def test_delete_nan_inq():
    # 测试径流数据反推处理功能
    cleaner = StreamflowBacktrack(
        "/ftproot/tests_stations_anomaly_detection/streamflow_backtrack/",
        "/ftproot/tests_stations_anomaly_detection/streamflow_backtrack/",
    )
    cleaner.delete_nan_inq(
        data_path="/ftproot/tests_stations_anomaly_detection/streamflow_backtrack/zq_CHN_songliao_21401550.csv",
        file="zq_CHN_songliao_21401550.csv",
        output_folder="/ftproot/tests_stations_anomaly_detection/streamflow_backtrack/",
    )
