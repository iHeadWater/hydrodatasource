"""
Author: liutiaxqabs 1498093445@qq.com
Date: 2024-05-28 10:24:16
LastEditors: Wenyu Ouyang
LastEditTime: 2025-01-14 09:26:36
FilePath: \hydrodatasource\scripts\china_rainfall_preprocessing.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
"""

import os
import sys
import pytest

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

sys.path.append(os.path.dirname(Path(os.path.abspath(__file__)).parent))
from const4scripts import RESULT_DIR, DATASET_DIR
from hydrodatasource.cleaner.rainfall_cleaner import RainfallCleaner, RainfallAnalyzer


def read_and_concat_csv(folder_path):
    """读取并合并文件夹下的所有 CSV 文件"""
    all_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.endswith(".csv")
    ]
    return pd.concat([pd.read_csv(f) for f in all_files], ignore_index=True)


def test_data_check_yearly():
    # 假设 df_era5land 和 df_station 已经加载为 DataFrame
    df_era5land = pd.read_csv(
        "/ftproot/era5land/songliao_2000_2024.csv"
    )  # 替换为真实路径
    df_station = read_and_concat_csv(
        "/ftproot/tests_stations_anomaly_detection/rainfall_cleaner/"
    )
    df_attr = pd.read_csv(
        "/ftproot/tests_stations_anomaly_detection/stations/stations.csv"
    )
    # 调用封装类中的方法并输出可信站点
    result_df = RainfallCleaner.data_check_yearly(
        df_era5land, df_station, df_attr, min_consecutive_years=1
    )
    print(result_df)
    result_df.to_csv("kexin.csv")
    pass


def test_data_check_hourly_extreme():
    station_lst = (
        pd.read_csv("kexin.csv")["STCD"].drop_duplicates().astype(str).unique()
    )
    data_df = read_and_concat_csv(
        "/ftproot/tests_stations_anomaly_detection/rainfall_cleaner/"
    )
    result_df = RainfallCleaner.data_check_hourly_extreme(
        data_df=data_df, station_lst=station_lst, climate_extreme_value=122
    )
    print(result_df)
    result_df.to_csv("extreme.csv")
    pass


def test_data_check_time_series():
    station_lst = (
        pd.read_csv("kexin.csv")["STCD"].drop_duplicates().astype(str).unique()
    )
    data_df = read_and_concat_csv(
        "/ftproot/tests_stations_anomaly_detection/rainfall_cleaner/"
    )
    result_df = RainfallCleaner.data_check_time_series(
        data_df=data_df,
        station_lst=station_lst,
        check_type="consistency",
        gradient_limit=120,
        window_size=24,
        consistent_value=0.5,
    )
    print(result_df)
    result_df.to_csv("consistency.csv")

    result_df = RainfallCleaner.data_check_time_series(
        data_df=data_df,
        station_lst=station_lst,
        check_type="gradient",
        gradient_limit=120,
        window_size=24,
        consistent_value=0.5,
    )
    print(result_df)
    result_df.to_csv("gradient.csv")
    pass


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
