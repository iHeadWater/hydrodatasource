"""
Author: liutiaxqabs 1498093445@qq.com
Date: 2024-05-28 10:24:16
LastEditors: Wenyu Ouyang
LastEditTime: 2025-01-15 10:17:44
FilePath: \hydrodatasource\scripts\china_rainfall_preprocessing.py
Description: test script for rainfall data preprocessing
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
# rainfall_cleaner.rainfall_clean("21401550")


# 测试泰森多边形平均值，碧流河为例。测试结果见 /ftproot/tests_stations_anomaly_detection/plot
basins_mean = RainfallAnalyzer(
    data_folder=rainfall_dir,
    output_folder=os.path.join(RESULT_DIR, "basins_rainfall_mean_available"),
    lower_bound=0,
    upper_bound=20000,
)
basins_mean.basins_polygon_mean(["21401550"])
