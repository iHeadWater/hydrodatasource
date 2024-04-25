"""
Author: liutiaxqabs 1498093445@qq.com
Date: 2024-04-19 14:00:27
LastEditors: liutiaxqabs 1498093445@qq.com
LastEditTime: 2024-04-24 13:48:19
FilePath: /hydrodatasource/hydrodatasource/cleaner/waterlevel_cleaner.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
"""

import numpy as np
import pandas as pd
from .cleaner import Cleaner


class WaterlevelCleaner(Cleaner):

    def __init__(self, data_path, grad_max=200, window_size=24, *args, **kwargs):
        self.temporal_list = pd.DataFrame()  # 初始化为空的DataFrame
        self.spatial_list = pd.DataFrame()
        self.grad_max = grad_max
        self.window_size = window_size
        super().__init__(data_path, *args, **kwargs)

    def rolling_fill(self, waterlevel_data):
        df = waterlevel_data.copy()
        # 计算滑动众数，如果众数不存在使用前一个有效值
        rolling_stat = (
            df["Z"]
            .rolling(window=self.window_size, min_periods=1)
            .apply(lambda x: x.iloc[-1] if x.mode().empty else x.mode()[0], raw=False)
        )
        # 使用计算得到的滑动统计值填补缺失值
        df["Z"] = df["Z"].fillna(rolling_stat)
        return df

    def moving_gradient_filter(self, waterlevel_data):
        # 创建数据副本以避免修改原始DataFrame
        df = waterlevel_data.copy()
        # 确保时间列 "TM" 是 datetime 类型
        df["TM"] = pd.to_datetime(df["TM"])

        # 计算水位变化梯度
        df["waterlevel_Change"] = df["Z"].diff()

        # 使用滑动窗口计算每个点的局部众数，众数不存在时使用中位数
        rolling_modes = (
            df["Z"]
            .rolling(window=self.window_size, min_periods=1)
            .apply(
                lambda x: x.median() if x.mode().empty else x.mode().iloc[0],
                raw=False,
            )
        )

        # 识别汛期
        df["Is_Flood_Season"] = df["TM"].apply(lambda x: 6 <= x.month <= 9)

        # 汛期与非汛期梯度阈值
        gradient_threshold_flood = self.grad_max
        gradient_threshold_non_flood = self.grad_max / 2

        # 替换汛期中梯度过高的数据点
        flood_mask = (df["Is_Flood_Season"] == True) & (
            df["waterlevel_Change"].abs() > gradient_threshold_flood
        )
        df.loc[flood_mask, "Z"] = None  # rolling_modes[flood_mask]

        # 替换非汛期中梯度过高的数据点
        non_flood_mask = (df["Is_Flood_Season"] == False) & (
            df["waterlevel_Change"].abs() > gradient_threshold_non_flood
        )
        df.loc[non_flood_mask, "Z"] = None  # rolling_modes[non_flood_mask]

        # 清理不需要的临时列
        df.drop(columns=["waterlevel_Change", "Is_Flood_Season"], inplace=True)

        return df

    def anomaly_process(self, methods=None):
        super().anomaly_process(methods)
        waterlevel_data = self.origin_df
        for method in methods:
            if method == "moving_grad":
                waterlevel_data = self.moving_gradient_filter(
                    waterlevel_data=waterlevel_data
                )
            if method == "roll":
                waterlevel_data = self.rolling_fill(
                    waterlevel_data=waterlevel_data
                )
            else:
                print("please check your method name")

        # self.processed_df["Z"] = waterlevel_data["Z"] # 最终结果赋值给processed_df
        # 新增一列进行存储
        self.processed_df[str(methods)] = waterlevel_data["Z"]
