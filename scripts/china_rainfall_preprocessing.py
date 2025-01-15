'''
Author: liutiaxqabs 1498093445@qq.com
Date: 2025-01-15 11:58:10
LastEditors: liutiaxqabs 1498093445@qq.com
LastEditTime: 2025-01-15 20:29:43
FilePath: /hydrodatasource/scripts/china_rainfall_preprocessing.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''


import os
import sys

from pathlib import Path

sys.path.append(os.path.dirname(Path(os.path.abspath(__file__)).parent))
from const4scripts import RESULT_DIR, DATASET_DIR
from hydrodatasource.cleaner.rainfall_cleaner import RainfallCleaner, RainfallAnalyzer
from hydrodatasource.reader.rainfall_reader import *


rainfall_dir = os.path.join(DATASET_DIR, "basins_songliao_pp_origin_available_data")
output_dir = os.path.join(RESULT_DIR, "basins_songliao_pp_stations")
station_attr =  os.path.join(rainfall_dir, "stations","stations.csv")

rainfall_cleaner = RainfallCleaner(rainfall_dir, output_dir)

rainfall_reader = RainfallReader(rainfall_dir, output_dir)
rainfall_reader.read_basin_rainfall("21401550")
rainfall_cleaner.rainfall_clean("21401550")


# 测试泰森多边形平均值，碧流河为例。测试结果见 /ftproot/tests_stations_anomaly_detection/plot
basins_mean = RainfallAnalyzer(
    data_folder=rainfall_dir,
    output_folder=os.path.join(RESULT_DIR, "basins_rainfall_mean_available"),
)
basins_mean.basins_polygon_mean(["21401550"])
