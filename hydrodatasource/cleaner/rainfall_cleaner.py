"""
Author: liutiaxqabs 1498093445@qq.com
Date: 2024-04-19 14:00:27
LastEditors: liutiaxqabs 1498093445@qq.com
LastEditTime: 2024-04-19 14:03:40
FilePath: /liutianxv1/hydrodatasource/hydrodatasource/cleaner/waterlevel_cleaner.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
"""

from .cleaner import Cleaner


class RainfallCleaner(Cleaner):
    def aggregate_data(self):
        # 实现聚合降雨数据的方法
        pass

    def identify_trends(self):
        # 实现识别降雨趋势的方法
        pass

    def extreme_filter(self):
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
            (df["Is_Flood_Season"] == False) & (df["DRP"] > extreme_value_non_flood),
            "DRP",
        ] = 0

        return df

    # 数据梯度筛查
    def gradient_filter(self):

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

    def anomaly_process(self, methods=None):
        super().process_data(methods)
        if "aggregate" in methods:
            self.aggregate_data()
        if "trends" in methods:
            self.identify_trends()
