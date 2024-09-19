'''
Author: liutiaxqabs 1498093445@qq.com
Date: 2024-05-28 10:24:16
LastEditors: liutiaxqabs 1498093445@qq.com
LastEditTime: 2024-06-22 10:40:23
FilePath: /hydrodatasource/tests/test_rainfall_cleaner.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

import pytest
from hydrodatasource.cleaner.rainfall_cleaner import RainfallCleaner, RainfallAnalyzer

import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm


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
        stations_csv_path="/ftproot/basins-origin/basins_pp_stations/20600340_stations.csv",# 站点表，其中ID列带有前缀‘pp_’
        shp_folder="/ftproot/basins-origin/basins_shp/20600340/",
        rainfall_data_folder="/ftproot/basins-origin/basins_songliao_pp_origin_data/20600340/",
        output_folder="/ftproot/basins-origin/basins_rainfall_mean/",
        output_log="/ftproot/basins-origin/basins_rainfall_mean/plot/summary_log.txt",
        output_plot="/ftproot/basins-origin/basins_rainfall_mean/plot/",
        lower_bound=200,
        upper_bound=2000,
    )
    basins_mean.basins_polygon_mean()


def test_basins_polygon_mean_folder():
    # 设置基础路径
    base_shp_folder = "/ftproot/basins-origin/basins_shp/"
    base_stations_csv_folder = "/ftproot/basins-origin/basins_pp_stations/"
    base_rainfall_data_folder = "/ftproot/basins-origin/basins_songliao_pp_origin_data/"
    output_folder = "/ftproot/basins-origin/basins_rainfall_mean/"
    output_log = os.path.join(output_folder, "plot", "summary_log.txt")
    output_plot = os.path.join(output_folder, "plot")

    # 获取shp_folder目录下的所有文件夹名称并排序
    subfolders = sorted([f.name for f in os.scandir(base_shp_folder) if f.is_dir()])

    # 遍历所有文件夹并运行 basins_polygon_mean
    for subfolder in tqdm(subfolders, desc="Processing basins"):
        stations_csv_path = os.path.join(base_stations_csv_folder, f"{subfolder}_stations.csv")
        shp_folder = os.path.join(base_shp_folder, subfolder)
        rainfall_data_folder = os.path.join(base_rainfall_data_folder, subfolder)
        
        if os.path.exists(stations_csv_path) and os.path.exists(shp_folder) and os.path.exists(rainfall_data_folder):
            try:
                basins_mean = RainfallAnalyzer(
                    stations_csv_path=stations_csv_path,
                    shp_folder=shp_folder,
                    rainfall_data_folder=rainfall_data_folder,
                    output_folder=output_folder,
                    output_log=output_log,
                    output_plot=output_plot,
                    lower_bound=200,
                    upper_bound=2000,
                )
                basins_mean.basins_polygon_mean()
            except Exception as e:
                print(f"Error processing {subfolder}: {e}")
                # 这里可以添加更多调试信息，比如打印 DataFrame 的列名
        else:
            print(f"Missing required files for {subfolder}")


def test_time_consistency():
    time_consistency_check = RainfallAnalyzer(
        rainfall_data_folder="/ftproot/tests_stations_anomaly_detection/rainfall_cleaner/",
        output_folder="/ftproot/tests_stations_anomaly_detection/results/",
    )
    time_consistency_check.time_consistency()
    
def test_spatial_consistency():
    basins_spatial = RainfallAnalyzer(
        stations_csv_path="/ftproot/tests_stations_anomaly_detection/stations/pp_stations.csv",
        shp_folder="/ftproot/tests_stations_anomaly_detection/shapefiles/",
        rainfall_data_folder="/ftproot/tests_stations_anomaly_detection/rainfall_cleaner/",
        output_folder="/ftproot/tests_stations_anomaly_detection/results/",
        output_plot="/ftproot/tests_stations_anomaly_detection/plot/",
    )
    basins_spatial.spatial_consistency()