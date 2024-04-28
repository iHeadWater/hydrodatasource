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
import xarray as xr
import os
from datetime import datetime, timedelta
from .cleaner import Cleaner


class RainfallCleaner(Cleaner):
    def __init__(
        self,
        data_path,
        era5_path,
        station_file,
        start_time,
        end_time,
        grad_max=200,
        extr_max=200,
        *args,
        **kwargs,
    ):
        self.temporal_list = pd.DataFrame()  # 初始化为空的DataFrame
        self.spatial_list = pd.DataFrame()
        self.grad_max = grad_max
        self.extr_max = extr_max
        self.era5_path = era5_path
        self.station_file = pd.read_csv(station_file, dtype={"STCD": str}) if isinstance(station_file, str) else station_file
        self.start_time, self.end_time = start_time, end_time
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

    def era5land_df(self, era5_path, start_time, end_time):
        output_dir = "/ftproot/era5land"
        output_file = os.path.join(output_dir, "tp.csv")
        
        # 检查是否已存在处理过的文件
        if os.path.exists(output_file):
            print("Using cached data from", output_file)
            return pd.read_csv(output_file)
        
        # 解析开始和结束时间
        start_date = datetime.strptime(start_time, "%Y-%m-%d")
        end_date = datetime.strptime(end_time, "%Y-%m-%d")

        # 初始化最终的 DataFrame
        final_df = pd.DataFrame()

        # 遍历目录中的所有文件
        for file_name in os.listdir(era5_path):
            if file_name.endswith(".nc"):  # 确保是 NetCDF 文件
                file_path = os.path.join(era5_path, file_name)

                # 打开 NetCDF 数据集
                try:
                    with xr.open_dataset(file_path) as ds:
                        # 提取并四舍五入经纬度数据
                        longitude = np.round(ds["longitude"].values, 1)
                        latitude = np.round(ds["latitude"].values, 1)
                        tp = ds["tp"]

                        # 选择数据集的第一个时间点，通常代表0点
                        tp_at_first_time = tp.isel(time=0)

                        # 获取时间信息
                        date_str = str(tp_at_first_time["time"].values)[:10]
                        data_date = datetime.strptime(date_str, "%Y-%m-%d")

                        # 检查是否在指定的时间范围内
                        if start_date <= data_date <= end_date:
                            tp_value = tp_at_first_time.values.flatten()

                            # 创建站点ID
                            station_ids = [
                                "era5land_{:.1f}_{:.1f}".format(lon, lat)
                                for lon in longitude
                                for lat in latitude
                            ]

                            # 创建临时 DataFrame
                            temp_df = pd.DataFrame(
                                {
                                    "ID": station_ids,
                                    "LON": np.repeat(longitude, len(latitude)),
                                    "LAT": np.tile(latitude, len(longitude)),
                                    "TP": tp_value,
                                    "TM": (data_date - timedelta(days=1)).strftime(
                                        "%Y-%m-%d"
                                    ),  # 使用前一天的日期
                                }
                            )

                            # 将临时 DataFrame 添加到最终 DataFrame
                            final_df = pd.concat([final_df, temp_df], ignore_index=True)
                except Exception as e:
                    print(f"Failed to process file {file_name}: {e}")

        # 空数据检查
        if final_df.empty:
            print("No data processed. Please check the input files and date range.")
            return None

        # 转换 TM 列为 datetime 类型，以便提取年份
        final_df["TM"] = pd.to_datetime(final_df["TM"])

        # 创建一个新的列 'Year' 来存储年份
        final_df["Year"] = final_df["TM"].dt.year

        # 按 ID 和 Year 分组，计算每个站点每年的 TP 总和
        annual_sum_df = (
            final_df.groupby(["ID", "Year"])
            .agg(
                {
                    "LON": "first",  # 取第一个经度作为代表
                    "LAT": "first",  # 取第一个纬度作为代表
                    "TP": "sum",  # 求和降水量
                }
            )
            .reset_index()
        )
        annual_sum_df.to_csv("/ftproot/era5land/tp.csv", index=False)
        return annual_sum_df

    # 空间信息筛选雨量站（ERA5-LAND校准）
    def spatial_era5land_detect(self, rainfall_data):
        # 截获起止时间计算era5land数据
        era5land_df = self.era5land_df(
            era5_path=self.era5_path, start_time=self.start_time, end_time=self.end_time
        )
        rainfall_df = self.sum_validate_detect(rainfall_data=rainfall_data)
        # 拿站点经纬度找最佳匹配得站点
        rainfall = pd.merge(rainfall_df, self.station_file, on='STCD', how='left')

        # 添加新列以存储匹配的ERA5 TP值
        rainfall['ERA5_TP'] = np.nan

        for index, rain_row in rainfall.iterrows():
            # 在ERA5数据中匹配网格
            matched = era5land_df[
                (era5land_df['LON'] <= rain_row['LON']) & 
                (era5land_df['LON'] + 0.1 > rain_row['LON']) & 
                (era5land_df['LAT'] - 0.1 < rain_row['LAT']) & 
                (era5land_df['LAT'] >= rain_row['LAT']) & 
                (era5land_df['Year'] == rain_row['Year'])
            ]
            
            if not matched.empty:
                # 如果找到匹配，取第一条匹配记录的TP值
                rainfall.at[index, 'ERA5_TP'] = matched.iloc[0]['TP']
            else:
                # 如果没有找到匹配，设置ERA5 TP值为NaN
                rainfall.at[index, 'ERA5_TP'] = np.nan

        # 判断合理性
        rainfall["IS_REA"] = False

        # 判断条件并设置 IS_REA 列的值
        valid_indices = (
            (rainfall["ERA5_TP"].notnull()) &  # ERA5_TP 不为空
            (rainfall["DRP_SUM"].notnull()) &  # DRP_SUM 不为空
            (0.8 * rainfall["ERA5_TP"] <= rainfall["DRP_SUM"]) &  # DRP_SUM 大于等于 0.8 * ERA5_TP
            (rainfall["DRP_SUM"] <= 1.2 * rainfall["ERA5_TP"])  # DRP_SUM 小于等于 1.2 * ERA5_TP
        )
        rainfall.loc[valid_indices, "IS_REA"] = True
        
        return rainfall[["STCD", "Year", "DRP_SUM", "LON", "LAT","ERA5_TP", "IS_REA"]]

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
