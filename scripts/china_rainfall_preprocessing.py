"""
Author: Wenyu Ouyang
Date: 2024-08-09 13:19:39
LastEditTime: 2025-01-15 21:25:09
LastEditors: Wenyu Ouyang
Description: Use this script to preprocess the rainfall data in China.
FilePath: \hydrodatasource\scripts\china_rainfall_preprocessing.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import os
import sys

from pathlib import Path

sys.path.append(os.path.dirname(Path(os.path.abspath(__file__)).parent))
from const4scripts import RESULT_DIR, DATASET_DIR
from hydrodatasource.cleaner.rainfall_cleaner import RainfallCleaner, RainfallAnalyzer


rainfall_dir = os.path.join(DATASET_DIR, "basins_songliao_pp_origin_available_data")
output_dir = os.path.join(RESULT_DIR, "basins_songliao_pp_stations")
rainfall_cleaner = RainfallCleaner(rainfall_dir, output_dir)

# 测试泰森多边形平均值，碧流河为例。测试结果见 /ftproot/tests_stations_anomaly_detection/plot
basins_mean = RainfallAnalyzer(
    data_folder=rainfall_dir,
    output_folder=os.path.join(RESULT_DIR, "basins_rainfall_mean_available"),
)
basins_mean.basins_polygon_mean(["21401550"])
