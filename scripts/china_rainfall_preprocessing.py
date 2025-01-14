"""
Author: liutiaxqabs 1498093445@qq.com
Date: 2024-05-28 10:24:16
LastEditors: Wenyu Ouyang
LastEditTime: 2025-01-14 20:47:13
FilePath: \hydrodatasource\scripts\china_rainfall_preprocessing.py
Description: test script for rainfall data preprocessing
"""

import os
import sys

import pandas as pd
from pathlib import Path
from tqdm import tqdm

sys.path.append(os.path.dirname(Path(os.path.abspath(__file__)).parent))
from const4scripts import RESULT_DIR, DATASET_DIR
from hydrodatasource.cleaner.rainfall_cleaner import RainfallCleaner, RainfallAnalyzer


rainfall_dir = os.path.join(DATASET_DIR, "basins_songliao_pp_origin_available_data")
output_dir = os.path.join(RESULT_DIR, "basins_songliao_pp_stations")
rainfall_cleaner = RainfallCleaner(rainfall_dir, output_dir)
# 调用封装类中的方法并输出可信站点
rainfall_cleaner.data_check_yearly(basin_id="21401550", min_consecutive_years=1)
print("check yearly data")

rainfall_cleaner.data_check_hourly_extreme(
    basin_id="21401550", climate_extreme_value=122
)
rainfall_cleaner.data_check_time_series(
    basin_id="21401550",
    check_type="consistency",
    gradient_limit=120,
    window_size=24,
    consistent_value=0.5,
)

rainfall_cleaner.data_check_time_series(
    basin_id="21401550",
    check_type="gradient",
    gradient_limit=120,
    window_size=24,
    consistent_value=0.5,
)



def test_basins_polygon_mean():
    # 测试泰森多边形平均值，碧流河为例。测试结果见 /ftproot/tests_stations_anomaly_detection/plot
    basins_mean = RainfallAnalyzer(
        stations_csv_path="/ftproot/basins-origin/basins_pp_stations/21401550_stations.csv",  # 站点表，其中ID列带有前缀‘pp_’
        shp_folder="/ftproot/basins-origin/basins_shp/21401550/",
        rainfall_data_folder="/ftproot/basins-origin/basins_songliao_pp_origin_available_data/21401550/",
        output_folder="/ftproot/basins-origin/basins_rainfall_mean_available/",
        output_log="/ftproot/basins-origin/basins_rainfall_mean_available/plot/summary_log.txt",
        output_plot="/ftproot/basins-origin/basins_rainfall_mean_available/plot/",
        lower_bound=0,
        upper_bound=20000,
    )
    basins_mean.basins_polygon_mean()


def test_basins_polygon_mean_folder():
    # 设置基础路径

    base_shp_folder = "/ftproot/basins-origin/basins_shp/"
    base_stations_csv_folder = "/ftproot/basins-origin/basins_pp_stations/"
    base_rainfall_data_folder = (
        "/ftproot/basins-origin/basins_songliao_pp_origin_available_data/"
    )
    output_folder = "/ftproot/basins-origin/basins_rainfall_mean_available/"
    output_log = os.path.join(output_folder, "plot", "summary_log.txt")
    output_plot = os.path.join(output_folder, "plot")

    # 获取shp_folder目录下的所有文件夹名称并排序
    subfolders = sorted(
        [f.name for f in os.scandir(base_rainfall_data_folder) if f.is_dir()]
    )

    # 遍历所有文件夹并运行 basins_polygon_mean
    for subfolder in tqdm(subfolders, desc="Processing basins"):
        stations_csv_path = os.path.join(
            base_stations_csv_folder, f"{subfolder}_stations.csv"
        )
        shp_folder = os.path.join(base_shp_folder, subfolder)
        rainfall_data_folder = os.path.join(base_rainfall_data_folder, subfolder)

        if (
            os.path.exists(stations_csv_path)
            and os.path.exists(shp_folder)
            and os.path.exists(rainfall_data_folder)
        ):
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
