"""
Author: liutiaxqabs 1498093445@qq.com
Date: 2024-04-19 14:00:06
LastEditors: liutiaxqabs 1498093445@qq.com
LastEditTime: 2024-04-22 17:56:18
FilePath: /hydrodatasource/hydrodatasource/cleaner/rainfall_cleaner.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
"""

import numpy as np
import pandas as pd
from .cleaner import Cleaner


class RainfallCleaner(Cleaner):
    def __init__(self, data_path, grad_max=200, extr_max=200, *args, **kwargs):
        self.temporal_list = pd.DataFrame()  # 初始化为空的DataFrame
        self.spatial_list = pd.DataFrame()
        self.grad_max = grad_max
        self.extr_max = extr_max
        super().__init__(data_path, *args, **kwargs)

    # 数据极大值检验
    def extreme_filter(self, rainfall_data):
        # 创建数据副本以避免修改原始DataFrame
        df = rainfall_data.copy()
        # 设置汛期与非汛期极值阈值
        extreme_value_flood = self.extr_max
        extreme_value_non_flood = self.extr_max / 2
        df["TM"] = pd.to_datetime(df["TM"])
        # 识别汛期
        df["Is_Flood_Season"] = df["TM"].apply(lambda x: 6 <= x.month <= 9)

        # 对超过极值阈值的数据进行处理，将DRP值设置为0
        df.loc[
            (df["Is_Flood_Season"] == True) & (df["DRP"] > extreme_value_flood),
            "DRP",
        ] = 0
        df.loc[
            (df["Is_Flood_Season"] == False) & (df["DRP"] > extreme_value_non_flood),
            "DRP",
        ] = 0

        return df

    # 数据梯度筛查
    def gradient_filter(self, rainfall_data):

        # 原始总降雨量
        original_total_rainfall = rainfall_data["DRP"].sum()

        # 创建数据副本以避免修改原始DataFrame
        df = rainfall_data.copy()

        # 计算降雨量变化梯度
        df["DRP_Change"] = df["DRP"].diff()

        # 汛期与非汛期梯度阈值
        gradient_threshold_flood = self.grad_max
        gradient_threshold_non_flood = self.grad_max / 2

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

        # 清理不再需要的列
        df.drop(columns=["DRP_Change", "Is_Flood_Season"], inplace=True)
        return df

    # 数据累计量检验
    def sum_validate_detect(self, rainfall_data):
        """
        检查每个站点每年的总降雨量是否在400到1200毫米之间，并为每个站点生成一个年度降雨汇总表。
        :param rainfall_data: 包含站点代码('STCD')、降雨量('DRP')和时间('TM')的DataFrame
        :return: 新的DataFrame，包含STCD, YEAR, DRP_SUM, IS_REA四列
        """
        # 复制数据并转换日期格式
        df = rainfall_data[
            [
                "STCD",
                "TM",
                "DRP",
            ]
        ].copy()
        df["TM"] = pd.to_datetime(df["TM"])
        df["Year"] = df["TM"].dt.year  # 添加年份列

        # 按站点代码和年份分组，并计算每年的累计降雨量
        grouped = df.groupby(["STCD", "Year"])
        annual_summary = grouped["DRP"].sum().reset_index(name="DRP_SUM")

        # 判断每年的累计降雨量是否在指定范围内
        annual_summary["IS_REA"] = annual_summary["DRP_SUM"].apply(
            lambda x: 400 <= x <= 1200
        )

        return annual_summary

    # 空间信息筛选雨量站（ERA5-LAND校准）
    def spatial_era5land_detect(self, rainfall_data):

        pass

    def anomaly_process(self, methods=None):
        super().anomaly_process(methods)
        rainfall_data = self.origin_df
        for method in methods:
            if method == "extreme":
                rainfall_data = self.extreme_filter(rainfall_data=rainfall_data)
            elif method == "gradient":
                rainfall_data = self.gradient_filter(rainfall_data=rainfall_data)
            elif method == "detect_sum":
                self.temporal_list = self.sum_validate_detect(
                    rainfall_data=rainfall_data
                )
            elif method == "detect_era5":
                self.spatial_list = self.spatial_era5land_detect(
                    rainfall_data=rainfall_data
                )
            else:
                print("please check your method name")

        # self.processed_df["DRP"] = rainfall_data["DRP"] # 最终结果赋值给processed_df
        # 新增一列进行存储
        self.processed_df[str(methods)] = rainfall_data["DRP"]
