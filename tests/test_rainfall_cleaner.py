"""
Author: liutiaxqabs 1498093445@qq.com
Date: 2024-04-22 18:02:00
LastEditors: liutiaxqabs 1498093445@qq.com
LastEditTime: 2024-05-31 11:34:41
FilePath: /hydrodatasource/tests/test_rainfall_cleaner.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE   
"""

import pytest
from hydrodatasource.cleaner.rainfall_cleaner import RainfallCleaner, RainfallAnalyzer

import pandas as pd
import matplotlib.pyplot as plt


def test_anomaly_process():
    # 测试降雨数据处理功能
    cleaner = RainfallCleaner(
        data_path="/ftproot/tests_stations_anomaly_detection/rainfall_cleaner/pp_CHN_songliao_21422982.csv",
        era5_path="/ftproot/tests_stations_anomaly_detection/era5land/",
        station_file="/ftproot/tests_stations_anomaly_detection/stations/pp_stations.csv",
        start_time="2020-01-01",
        end_time="2022-10-07",
    )
    # methods默认可以联合调用，也可以单独调用。大多数情况下，默认调用detect_sum
    methods = ["detect_sum"]
    cleaner.anomaly_process(methods)

    print(cleaner.origin_df)
    print(cleaner.processed_df)
    cleaner.processed_df.to_csv(
        "/ftproot/tests_stations_anomaly_detection/results/sampledatatest.csv"
    )
    cleaner.temporal_list.to_csv(
        "/ftproot/tests_stations_anomaly_detection/results/temporal_list.csv"
    )
    cleaner.spatial_list.to_csv(
        "/ftproot/tests_stations_anomaly_detection/results/spatial_list.csv"
    )


def test_basins_polygon_mean():
    # 测试泰森多边形平均值，碧流河为例。测试结果见 /ftproot/tests_stations_anomaly_detection/plot
    basins_mean = RainfallAnalyzer(
        stations_csv_path="/ftproot/tests_stations_anomaly_detection/stations/pp_stations_.csv",# 站点表，其中ID列带有前缀‘pp_’
        shp_folder="/ftproot/tests_stations_anomaly_detection/shapefiles/",
        rainfall_data_folder="/ftproot/tests_stations_anomaly_detection/rainfall_cleaner/",
        output_folder="/ftproot/tests_stations_anomaly_detection/basins_rainfall_mean/",
        output_log="/ftproot/tests_stations_anomaly_detection/plot/summary_log.txt",
        output_plot="/ftproot/tests_stations_anomaly_detection/plot/",
        lower_bound=200,
        upper_bound=2000,
    )
    basins_mean.basins_polygon_mean()

