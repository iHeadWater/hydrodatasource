'''
Author: liutiaxqabs 1498093445@qq.com
Date: 2024-04-19 14:00:06
LastEditors: liutiaxqabs 1498093445@qq.com
LastEditTime: 2024-06-21 12:13:04
FilePath: /hydrodatasource/hydrodatasource/cleaner/rainfall_cleaner.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

import numpy as np
import pandas as pd
import xarray as xr
import os
from datetime import datetime, timedelta
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
from scipy.spatial import Voronoi
import numpy as np
from geopandas.tools import sjoin
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
        self.station_file = (
            pd.read_csv(station_file, dtype={"STCD": str})
            if isinstance(station_file, str)
            else station_file
        )
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
        检查每个站点每年的总降雨量是否在200到2000毫米之间，并为每个站点生成一个年度降雨汇总表。
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
            lambda x: 200 <= x <= 2000
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
        rainfall = pd.merge(rainfall_df, self.station_file, on="STCD", how="left")

        # 添加新列以存储匹配的ERA5 TP值
        rainfall["ERA5_TP"] = np.nan

        for index, rain_row in rainfall.iterrows():
            # 在ERA5数据中匹配网格
            matched = era5land_df[
                (era5land_df["LON"] <= rain_row["LON"])
                & (era5land_df["LON"] + 0.1 > rain_row["LON"])
                & (era5land_df["LAT"] - 0.1 < rain_row["LAT"])
                & (era5land_df["LAT"] >= rain_row["LAT"])
                & (era5land_df["Year"] == rain_row["Year"])
            ]

            if not matched.empty:
                # 如果找到匹配，取第一条匹配记录的TP值
                rainfall.at[index, "ERA5_TP"] = matched.iloc[0]["TP"]
            else:
                # 如果没有找到匹配，设置ERA5 TP值为NaN
                rainfall.at[index, "ERA5_TP"] = np.nan

        # 判断合理性
        rainfall["IS_REA"] = False

        # 判断条件并设置 IS_REA 列的值
        valid_indices = (
            (rainfall["ERA5_TP"].notnull())  # ERA5_TP 不为空
            & (rainfall["DRP_SUM"].notnull())  # DRP_SUM 不为空
            & (
                0.8 * rainfall["ERA5_TP"] <= rainfall["DRP_SUM"]
            )  # DRP_SUM 大于等于 0.8 * ERA5_TP
            & (
                rainfall["DRP_SUM"] <= 1.2 * rainfall["ERA5_TP"]
            )  # DRP_SUM 小于等于 1.2 * ERA5_TP
        )
        rainfall.loc[valid_indices, "IS_REA"] = True

        return rainfall[["STCD", "Year", "DRP_SUM", "LON", "LAT", "ERA5_TP", "IS_REA"]]

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
        self.processed_df[methods[0]] = rainfall_data["DRP"]


class RainfallAnalyzer:
    def __init__(
        self,
        stations_csv_path=None,
        shp_folder=None,
        rainfall_data_folder=None,
        output_folder=None,
        output_log=None,
        output_plot=None,
        lower_bound=None,
        upper_bound=None,
    ):
        self.stations_csv_path = stations_csv_path
        self.shp_folder = shp_folder
        self.rainfall_data_folder = rainfall_data_folder
        self.output_folder = output_folder
        self.output_log = output_log
        self.output_plot = output_plot
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def filter_and_save_csv(self):
        """
        筛选降雨数据，根据每年的总降雨量（DRP）进行过滤，保留符合最低和最高阈值的数据。

        参数：
        input_folder - 包含降雨数据的文件夹路径。
        lower_bound - 降雨量最低阈值。
        upper_bound - 降雨量最高阈值。

        返回：
        过滤后的降雨数据DataFrame。
        """
        print("Filtering data by yearly total DRP")
        input_folder = self.rainfall_data_folder
        filtered_data_list = []
        for file in os.listdir(input_folder):
            if file.endswith(".csv"):
                file_path = os.path.join(input_folder, file)
                data = pd.read_csv(file_path)
                data["TM"] = pd.to_datetime(data["TM"], errors='coerce')
                #data["TM"] = pd.to_datetime(data["TM"], format="%Y-%m-%d %H:%M:%S.%f", errors='coerce')
                data["DRP"] = data["DRP"].astype(float)

                data["ID"] = file.replace(".csv", "")
                for year, group in data.groupby(data["TM"].dt.year):
                    drp_sum = group["DRP"].sum()
                    if self.lower_bound <= drp_sum <= self.upper_bound:
                        # print(f"File {file} contains valid data for year {year} with DRP sum {drp_sum}")
                        filtered_data_list.append(group)
        if filtered_data_list:
            return pd.concat(filtered_data_list, ignore_index=True)
        else:
            return pd.DataFrame()

    def read_data(self, basin_shp_path):
        """
        读取站点信息和流域shapefile数据。

        参数：
        stations_csv_path - 站点信息CSV文件路径。
        basin_shp_path - 流域shapefile文件路径。

        返回：
        stations_df - 站点信息DataFrame。
        basin - 流域shapefile的GeoDataFrame。
        """
        stations_df = pd.read_csv(self.stations_csv_path)
        stations_df.dropna(subset=["LON", "LAT"], inplace=True)
        basin = gpd.read_file(basin_shp_path)
        return stations_df, basin

    def process_stations(self, stations_df, basin):
        """
        筛选位于流域内部的站点数据。

        参数：
        stations_df - 站点信息DataFrame。
        basin - 流域shapefile的GeoDataFrame。

        返回：
        stations_within_basin - 位于流域内部的站点GeoDataFrame。
        """
        print("Processing stations within the basin")
        gdf_stations = gpd.GeoDataFrame(
            stations_df,
            geometry=[Point(xy) for xy in zip(stations_df.LON, stations_df.LAT)],
            crs="EPSG:4326",
        )
        gdf_stations = gdf_stations.to_crs(basin.crs)
        stations_within_basin = sjoin(gdf_stations, basin, predicate="within")
        print(f"Found {len(stations_within_basin)} stations within the basin")
        return stations_within_basin

    def calculate_voronoi_polygons(self, stations, basin):
        """
        计算泰森多边形并裁剪至流域边界。

        参数：
        stations - 位于流域内部的站点GeoDataFrame。
        basin - 流域shapefile的GeoDataFrame。

        返回：
        clipped_polygons - 裁剪后的泰森多边形GeoDataFrame。
        """
        if len(stations) < 2:
            stations["original_area"] = np.nan
            stations["clipped_area"] = np.nan
            stations["area_ratio"] = 1.0
            return stations

        # 获取流域边界的最小和最大坐标，构建边界框
        x_min, y_min, x_max, y_max = basin.total_bounds

        # 扩展边界框
        x_min -= 1.0 * (x_max - x_min)
        x_max += 1.0 * (x_max - x_min)
        y_min -= 1.0 * (y_max - y_min)
        y_max += 1.0 * (y_max - y_min)

        bounding_box = np.array(
            [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
        )

        # 提取站点坐标
        points = np.array([point.coords[0] for point in stations.geometry])

        # 将站点坐标与边界框点结合，确保Voronoi多边形覆盖整个流域
        points_extended = np.concatenate((points, bounding_box), axis=0)

        # 计算Voronoi图
        vor = Voronoi(points_extended)

        # 提取每个点对应的Voronoi区域
        regions = [vor.regions[vor.point_region[i]] for i in range(len(points))]

        # 生成多边形
        polygons = [
            Polygon([vor.vertices[i] for i in region if i != -1])
            for region in regions
            if -1 not in region
        ]

        # 创建GeoDataFrame
        gdf_polygons = gpd.GeoDataFrame(geometry=polygons, crs=stations.crs)
        gdf_polygons["STCD"] = stations["STCD"].values
        gdf_polygons["original_area"] = gdf_polygons.geometry.area

        # 计算流域的总面积
        basin_area = basin.geometry.area.sum()
        print(f"Basin area: {basin_area}")

        # 计算原始泰森多边形的总面积
        total_original_area = gdf_polygons["original_area"].sum()
        print(f"Total original Voronoi polygons area: {total_original_area}")

        # 将多边形裁剪到流域边界
        clipped_polygons = gpd.clip(gdf_polygons, basin)
        clipped_polygons["clipped_area"] = clipped_polygons.geometry.area
        clipped_polygons["area_ratio"] = (
            clipped_polygons["clipped_area"] / clipped_polygons["clipped_area"].sum()
        )

        # 计算裁剪后泰森多边形的总面积
        total_clipped_area = clipped_polygons["clipped_area"].sum()
        print(f"Total clipped Voronoi polygons area: {total_clipped_area}")

        # 打印年度数据汇总并将其追加到日志文件中
        log_file = self.output_log
        with open(log_file, "a") as f:
            log_entries = [
                f"Basin area: {basin_area}",
                f"Total original Voronoi polygons area: {total_original_area}",
                f"Total clipped Voronoi polygons area: {total_clipped_area}",
            ]
            for entry in log_entries:
                print(entry)
                f.write(entry + "\n")

        return clipped_polygons

    def calculate_weighted_rainfall(self, thiesen_polygons, rainfall_df):
        """
        计算加权平均降雨量。

        参数：
        thiesen_polygons - 泰森多边形GeoDataFrame。
        rainfall_df - 降雨数据DataFrame。

        返回：
        weighted_average_rainfall - 加权平均降雨量DataFrame。
        """
        thiesen_polygons["STCD"] = thiesen_polygons["STCD"].astype(str)
        rainfall_df["STCD"] = rainfall_df["STCD"].astype(str)

        # 合并泰森多边形和降雨数据
        merged_data = pd.merge(thiesen_polygons, rainfall_df, on="STCD")

        # 计算加权降雨量
        merged_data["weighted_rainfall"] = (
            merged_data["DRP"] * merged_data["area_ratio"]
        )

        # 按时间分组并计算加权平均降雨量
        weighted_average_rainfall = (
            merged_data.groupby("TM")["weighted_rainfall"].sum().reset_index()
        )

        return weighted_average_rainfall

    def display_results(
        self,
        year,
        valid_stations,
        thiesen_polygons_year,
        yearly_data,
        average_rainfall,
        basin,
    ):
        """
        显示处理结果，包括地图展示、站点信息、降雨量信息和平均降雨量。

        参数：
        year - 当前处理的年份。
        valid_stations - 符合条件的站点GeoDataFrame。
        yearly_data - 当前年份的降雨数据DataFrame。
        average_rainfall - 加权平均降雨量DataFrame。
        basin - 流域shapefile的GeoDataFrame。
        """
        print(f"Displaying results for year {year}")

        # 绘制经纬度图像
        fig, ax = plt.subplots(figsize=(10, 10))
        basin.plot(ax=ax, color="lightgrey", edgecolor="black")
        thiesen_polygons_year.plot(
            ax=ax, facecolor="blue", edgecolor="black", markersize=50
        )
        valid_stations.plot(ax=ax, color="red", markersize=50)
        plt.title(f"Stations within basin {basin['BASIN_ID'].iloc[0]} for year {year}")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        # 生成文件名
        file_name = f"{basin['BASIN_ID'].iloc[0]}_{year}.png"
        file_path = f"{self.output_plot}/{file_name}"
        # 保存图像
        plt.savefig(file_path)
        # plt.show()

        # 输出站点名称和数量
        station_names = valid_stations["ID"].tolist()
        station_count = len(station_names)
        print(f"Stations for year {year}: {station_names}")
        print(f"Total number of stations: {station_count}")

        # 输出对应年份的数据
        filtered_yearly_data = yearly_data[yearly_data["ID"].isin(station_names)]

        yearly_summary = (
            filtered_yearly_data.groupby("ID")
            .agg({"STCD": "first", "DRP": "sum"})
            .reset_index()
        )
        print(f"Yearly data summary for year {year}:\n{yearly_summary}")

        # 输出平均雨量数据
        mean_rainfall = average_rainfall["mean_rainfall"].sum()
        print(f"Average rainfall for year {year}: {mean_rainfall}")

        # 追加日志
        # 打印年度数据汇总并将其追加到日志文件中
        log_file = self.output_log
        with open(log_file, "a") as f:
            log_entries = [
                f"BASINS: {basin['BASIN_ID'].iloc[0]}",
                f"Displaying results for year {year}",
                f"Stations for year {year}: {station_names}",
                f"Total number of stations: {station_count}",
                f"Yearly data summary for year {year}:\n{yearly_summary}",
                f"Average rainfall for year {year}: {mean_rainfall}\n",
            ]
            for entry in log_entries:
                print(entry)
                f.write(entry + "\n")

    def process_basin(self, basin_shp_path, filtered_data):
        """
        处理每个流域的降雨数据，计算泰森多边形和面平均降雨量。

        参数：
        basin_shp_path - 流域shapefile文件路径。
        stations_csv_path - 站点信息CSV文件路径。
        filtered_data - 预先过滤的降雨数据DataFrame。
        output_folder - 输出文件夹路径。
        """
        all_years_rainfall = []
        stations_df, basin = self.read_data(basin_shp_path)

        years = filtered_data["TM"].dt.year.unique()

        for year in sorted(years):
            print(
                f"Processing basin {os.path.basename(basin_shp_path)} for year {year}"
            )
            # 打印年度数据汇总并将其追加到日志文件中
            log_file = self.output_log
            with open(log_file, "a") as f:
                f.write(
                    f"Processing basin {os.path.basename(basin_shp_path)} for year {year}\n"
                )
            yearly_data = filtered_data[filtered_data["TM"].dt.year == year]

            if yearly_data.empty:
                print(
                    f"No valid data for basin {os.path.basename(basin_shp_path)} in year {year}"
                )
                # 打印年度数据汇总并将其追加到日志文件中
                log_file = self.output_log
                with open(log_file, "a") as f:
                    f.write(
                        f"No valid stations for basin {os.path.basename(basin_shp_path)} in year {year}\n"
                    )
                continue

            # 筛选符合条件的每年站点数据
            yearly_stations = yearly_data["ID"].unique()
            valid_stations = self.process_stations(stations_df, basin)
            valid_stations = valid_stations[valid_stations["ID"].isin(yearly_stations)]

            if valid_stations.empty:
                print(
                    f"No valid stations for basin {os.path.basename(basin_shp_path)} in year {year}"
                )
                # 打印年度数据汇总并将其追加到日志文件中
                log_file = self.output_log
                with open(log_file, "a") as f:
                    f.write(
                        f"No valid stations for basin {os.path.basename(basin_shp_path)} in year {year}\n"
                    )

                continue

            thiesen_polygons_year = self.calculate_voronoi_polygons(
                valid_stations, basin
            )
            average_rainfall = self.calculate_weighted_rainfall(
                thiesen_polygons_year, yearly_data
            )
            average_rainfall.columns = ["TM", "mean_rainfall"]
            basin_id = os.path.splitext(os.path.basename(basin_shp_path))[0]
            average_rainfall["ID"] = basin_id
            all_years_rainfall.append(average_rainfall)

            # 调用展示函数
            self.display_results(
                year,
                valid_stations,
                thiesen_polygons_year,
                yearly_data,
                average_rainfall,
                basin,
            )

        if all_years_rainfall:
            final_result = pd.concat(all_years_rainfall, ignore_index=True)
            basin_output_folder = os.path.join(self.output_folder, basin_id)
            os.makedirs(basin_output_folder, exist_ok=True)
            output_file = os.path.join(basin_output_folder, f"{basin_id}_rainfall.csv")
            final_result.to_csv(output_file, index=False)
            print(f"Result for basin {basin_id} saved to {output_file}")
        else:
            print(
                f"No valid data for basin {os.path.splitext(os.path.basename(basin_shp_path))[0]}"
            )

    def basins_polygon_mean(self):
        """
        主函数，执行整个数据处理流程。

        参数：
        stations_csv_path - 站点信息CSV文件路径。
        shp_folder - 流域shapefile文件夹路径。
        rainfall_data_folder - 降雨数据文件夹路径。
        lower_bound - 降雨量最低阈值。
        upper_bound - 降雨量最高阈值。
        output_folder - 输出文件夹路径。
        """
        # 先筛选降雨数据，保留符合最低阈值的数据
        filtered_data = self.filter_and_save_csv()
        for shp_file in os.listdir(self.shp_folder):
            if shp_file.endswith(".shp"):
                basin_shp_path = os.path.join(self.shp_folder, shp_file)
                self.process_basin(basin_shp_path, filtered_data)

    # 添加时间一致性检验功能
    def check_time_consistency(self, df, hours=24):
        # 假设时间列为'TM'和降雨量列为'DRP'
        datetime_col = "TM"
        rainfall_col = "DRP"

        # 解析日期时间列
        df[datetime_col] = pd.to_datetime(df[datetime_col])

        # 按时间排序
        df = df.sort_values(by=datetime_col)

        # 添加一个布尔列来标记异常
        df["is_anomaly"] = False

        # 检查是否有连续24小时降雨量完全一致且非零的情况
        for i in range(len(df) - hours + 1):
            window = df.iloc[i : i + hours]
            # 过滤掉NaN值
            if window[rainfall_col].isna().sum() == 0:
                if (
                    len(window[rainfall_col].unique()) == 1
                    and window[rainfall_col].iloc[0] != 0
                ):
                    df.loc[window.index, "is_anomaly"] = True

        return df

    def time_consistency(self):
        all_anomalies = pd.DataFrame()

        # 遍历rainfall_data_folder中的所有文件
        for file_name in os.listdir(self.rainfall_data_folder):
            file_path = os.path.join(self.rainfall_data_folder, file_name)

            if os.path.isfile(file_path) and file_name.endswith(".csv"):
                # 读取CSV文件
                df = pd.read_csv(file_path)

                # 检查时间一致性
                df = self.check_time_consistency(df)

                # 提取标记为异常的记录
                anomalies = df[df["is_anomaly"]]

                # 将异常数据添加到总的DataFrame中
                all_anomalies = pd.concat([all_anomalies, anomalies], ignore_index=True)

        # 将所有异常数据保存到一个txt文件中
        output_file = os.path.join(self.output_folder, "time_consistency_anomalies.txt")
        with open(output_file, "w") as f:
            f.write(all_anomalies.to_string(index=False))

        print(f"异常数据已保存到 {output_file}")

    # 添加空间一致性检验功能
    def spatial_consistency(self):
        """
        主函数，执行整个数据处理流程。

        参数：
        stations_csv_path - 站点信息CSV文件路径。
        shp_folder - 流域shapefile文件夹路径。
        rainfall_data_folder - 降雨数据文件夹路径。
        output_folder - 输出文件夹路径。
        """
        self.lower_bound = -1
        self.upper_bound = 50000
        filtered_data = self.filter_and_save_csv()
        for shp_file in os.listdir(self.shp_folder):
            if shp_file.endswith(".shp"):
                basin_shp_path = os.path.join(self.shp_folder, shp_file)
                #########################################################
                # 以下是空间一致性检验的代码
                #########################################################
