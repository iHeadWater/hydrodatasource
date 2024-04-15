"""
Author: Wenyu Ouyang
Date: 2024-04-08 09:52:32
LastEditTime: 2024-04-09 17:57:09
LastEditors: Wenyu Ouyang
Description: Anomaly process for rainfall data
FilePath: \hydrodatasource\hydrodatasource\cleaner\rain_anomaly.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import xarray as xr
import pandas as pd
import numpy as np
import os


# 数据极值筛查
def rainfall_extreme_filter(df):
    """
    根据汛期和非汛期的极值阈值筛查DataFrame中的降雨数据，超过极值的数据将其DRP值设置为0。

    参数:
    - df: 包含降雨数据的DataFrame。

    返回:
    - 经过极值筛查处理后的DataFrame。
    """
    # 设置汛期与非汛期极值阈值
    extreme_value_flood = 200
    extreme_value_non_flood = 50
    df["TM"] = pd.to_datetime(df["TM"])
    # 识别汛期
    df["Is_Flood_Season"] = df["TM"].apply(lambda x: 6 <= x.month <= 9)

    # 对超过极值阈值的数据进行处理，将DRP值设置为0
    df.loc[
        (df["Is_Flood_Season"] == True) & (df["DRP"] > extreme_value_flood), "DRP"
    ] = 0
    df.loc[
        (df["Is_Flood_Season"] == False) & (df["DRP"] > extreme_value_non_flood), "DRP"
    ] = 0

    return df


# 数据梯度筛查
def rainfall_gradient_filter(df):
    """
    处理DataFrame中的降雨数据，考虑汛期和非汛期的不同梯度阈值。

    参数:
    - df: 包含降雨数据的DataFrame。

    返回:
    - 处理后的DataFrame。
    """
    # 原始总降雨量
    original_total_rainfall = df["DRP"].sum()

    # 计算降雨量变化梯度
    df["DRP_Change"] = df["DRP"].diff()

    # 汛期与非汛期梯度阈值
    gradient_threshold_flood = 20
    gradient_threshold_non_flood = 10

    # 识别汛期
    df["TM"] = pd.to_datetime(df["TM"])
    df["Is_Flood_Season"] = df["TM"].apply(lambda x: 6 <= x.month <= 9)

    # 处理异常值
    df.loc[
        (df["Is_Flood_Season"] == True)
        & (df["DRP_Change"].abs() > gradient_threshold_flood),
        "DRP",
    ] = 0
    df.loc[
        (df["Is_Flood_Season"] == False)
        & (df["DRP_Change"].abs() > gradient_threshold_non_flood),
        "DRP",
    ] = 0

    # 调整后的总降雨量
    adjusted_total_rainfall = df["DRP"].sum()

    # 打印数据总量的变化
    print(f"Original Total Rainfall: {original_total_rainfall} mm")
    print(f"Adjusted Total Rainfall: {adjusted_total_rainfall} mm")
    print(f"Change: {adjusted_total_rainfall - original_total_rainfall} mm")

    return df


# 空间数据对比


# 数据格式规范
def rainfall_format_normalization(df, freq="h"):
    # 转换时间列为datetime类型
    df["TM"] = pd.to_datetime(df["TM"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    # 尝试转换STCD列为整数，如果失败，则转换为字符串
    try:
        df["STCD"] = df["STCD"].astype(int).astype(str)
    except ValueError:
        df["STCD"] = df["STCD"].astype(str)

    # 生成完整的时间序列，确保没有间隔
    full_time_series = pd.date_range(start=df["TM"].min(), end=df["TM"].max(), freq="h")
    full_df = pd.DataFrame(full_time_series, columns=["TM"])

    # 确保合并前两个DataFrame的时间列数据类型一致
    df["TM"] = pd.to_datetime(df["TM"])
    full_df["TM"] = pd.to_datetime(full_df["TM"])

    # 合并原始数据到完整的时间序列中
    df_complete = pd.merge(full_df, df, on="TM", how="left")

    # 插补缺失数据
    stcd_fill_value = None if df["STCD"].dropna().empty else df["STCD"].dropna().iloc[0]
    df_complete["DRP"].fillna(0, inplace=True)
    df_complete["STCD"].fillna(stcd_fill_value, inplace=True)

    return df_complete
