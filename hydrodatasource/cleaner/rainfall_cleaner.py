"""
Author: liutiaxqabs 1498093445@qq.com
Date: 2024-04-19 14:00:06
LastEditors: Wenyu Ouyang
LastEditTime: 2025-01-15 11:58:32
FilePath: \hydrodatasource\hydrodatasource\cleaner\rainfall_cleaner.py
Description: data preprocessing for station gauged rainfall data
"""

import collections
import logging
import numpy as np
import pandas as pd
import os
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
from scipy.spatial import Voronoi
from geopandas.tools import sjoin
from tqdm import tqdm
from hydrodatasource.cleaner.cleaner import Cleaner

class RainfallCleaner(Cleaner):
    def __init__(self, data_folder, output_folder):
        """All files to be cleaned are in the data_dir

        Parameters
        ----------
        data_dir : _type_
            _description_
        """
        self.data_folder = data_folder
        self.output_folder = output_folder
        self.data_source_description = self.set_data_source_describe()
        self._check_file_format()
        self.station_info = self.read_site_info()

    def set_data_source_describe(self):
        data_source_dir = self.data_folder
        # we must have a file to provide the reservoir basic information
        era5land_file = os.path.join(data_source_dir, "songliao_2000_2024.csv")
        station_info_file = os.path.join(data_source_dir, "stations", "stations.csv")
    
        return collections.OrderedDict(
            REANALYSIS_FILE=era5land_file, STATIONS_INFO_FILE=station_info_file
        )

    def _check_file_format(self):
        # check if the file format is correct
        pass

    def read_site_info(self):
        station_info_file = self.data_source_description["STATIONS_INFO_FILE"]
        return pd.read_csv(station_info_file)

    def read_and_concat_csv(self, basin_id):
        """读取并合并文件夹下的所有 CSV 文件"""
        folder_path = os.path.join(self.data_folder, basin_id)
        all_files = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.endswith(".csv")
        ]
        return pd.concat([pd.read_csv(f) for f in all_files], ignore_index=True)

    def data_check_yearly(
        self,
        basin_id,
        year_range=None,
        diff_range=None,
        min_true_percentage=0.75,
        min_consecutive_years=3,
        modify = False
    ):
        """
        计算遥感数据与站点数据之间的降水差异，评估站点可靠性，并返回可信任的站点列表。

        参数:
        ----------
        basin_id : str
            Basin ID
        year_range : list, 可选
            要筛选的年份范围，默认是 [2010, 2024]。
        diff_range : list, 可选
            站点数据和遥感数据之间的ratio差异范围
            0.5 means station data is 0.5 times of reanalysis data
            2.0 means station data is 2 times of reanalysis data
        min_true_percentage : float, 可选
            要求可信年份的最小比例，默认 0.75。
        min_consecutive_years : int, 可选
            最小连续可信年份数，默认 3。

        返回:
        ------- 
        result_df : pd.DataFrame
            可信站点的 DataFrame，包含 'STCD'、'Latitude'、'Longitude' 和 'Reason' 列。
        """
        if year_range is None:
            year_range = [2010, 2024]
        if diff_range is None:
            diff_range = [0.4, 2.5]
        df_attr = self.station_info
        # 包含遥感数据（era5land）的降水数据
        df_era5land = pd.read_csv(self.data_source_description["REANALYSIS_FILE"])
        # 包含站点降水数据的 DataFrame
        df_station = self.read_and_concat_csv(basin_id)
        # 提取年份并处理日期格式不一致的问题
        df_station = self._station_yearly_sum(df_attr, df_station)
        output_dir = os.path.join(self.output_folder, f"{basin_id}")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 筛选年份范围
        df_era5land = df_era5land[
            (df_era5land["year"] >= year_range[0])
            & (df_era5land["year"] <= year_range[1])
        ]
        df_station = df_station[
            (df_station["Year"] >= year_range[0])
            & (df_station["Year"] <= year_range[1])
        ]

        # 将经纬度精度保留到1位小数
        df_station["LTTD"] = df_station["LTTD"].round(1)
        df_station["LGTD"] = df_station["LGTD"].round(1)
        df_era5land["latitude"] = df_era5land["latitude"].round(1)
        df_era5land["longitude"] = df_era5land["longitude"].round(1)
        df_era5land["total_precipitation"] = df_era5land["total_precipitation"] * 1000

        # 用于保存可信站点的结果
        results = []
        trusted_stations = []

        # 遍历站点数据
        for stcd, group in df_station.groupby("STCD"):
            group = group.sort_values("Year")

            # 获取站点的经纬度
            lat = group["LTTD"].values[0]
            lon = group["LGTD"].values[0]

            # 检查该站点是否存在遥感数据
            matched_era5land = df_era5land[
                (df_era5land["latitude"] == lat) & (df_era5land["longitude"] == lon)
            ]

            if not matched_era5land.empty:
                # 获取匹配的遥感数据
                reliable_years, the_result = self._reliable_years(
                    diff_range, stcd, group, lat, lon, matched_era5land
                )
                # append list to list
                results = results + the_result

                if reason := self._get_trust_reason(
                    reliable_years, min_true_percentage, min_consecutive_years
                ):
                    # 保存该站点为可信站点
                    trusted_stations.append(
                        {
                            "STCD": stcd,
                            "Latitude": lat,
                            "Longitude": lon,
                            "Reason": reason,
                        }
                    )

        # 转换详细结果为 DataFrame 并保存
        detailed_results_df = pd.DataFrame(results)
        detailed_results_df = detailed_results_df.drop_duplicates()
        detailed_results_df.to_csv(os.path.join(output_dir, "detaildata.csv"))

        # 转换可信站点结果为 DataFrame 并保存
        trusted_stations_df = pd.DataFrame(trusted_stations)

        # 排序并返回结果
        trusted_stations_df["STCD"] = trusted_stations_df["STCD"].astype(str)
        trusted_stations_df = trusted_stations_df.sort_values(by="STCD")
        trusted_stations_df.to_csv(os.path.join(output_dir, "kexin.csv"))

        if modify == True:
            trusted_stcd_list = trusted_stations_df["STCD"].tolist()
            folder_path = os.path.join(self.data_folder, basin_id)
            [
                os.remove(os.path.join(folder_path, file))
                for file in os.listdir(folder_path)
                if not set(pd.read_csv(os.path.join(folder_path, file))["STCD"].astype(str).unique()).intersection(trusted_stcd_list)
            ]

        return trusted_stations_df

    def _get_trust_reason(
        self, reliable_years, min_true_percentage, min_consecutive_years
    ):
        true_years = sum(r is True for r in reliable_years)
        total_years = len(reliable_years)
        true_percentage = true_years / total_years

        reason = None
        if true_percentage >= min_true_percentage:
            reason = f"{int(min_true_percentage * 100)}% 以上年份为 True"

            # 判断是否有连续的可信年份
        if total_years >= min_consecutive_years:
            consecutive_true = any(
                sum(reliable_years[i : i + min_consecutive_years])
                >= min_consecutive_years
                for i in range(len(reliable_years) - min_consecutive_years + 1)
            )
            if consecutive_true:
                if reason is not None:
                    reason += f", 连续 {min_consecutive_years} 年为 True"
                else:
                    reason = f"连续 {min_consecutive_years} 年为 True"
        return reason

    def _reliable_years(self, diff_range, stcd, group, lat, lon, matched_era5land):
        reliable_years = []
        results_ = []
        for year in group["Year"]:
            station_rainfall = group[group["Year"] == year]["DRP"].values[0]
            remote_precipitation = matched_era5land[matched_era5land["year"] == year][
                "total_precipitation"
            ].values

            if remote_precipitation.size > 0:
                remote_precipitation = remote_precipitation[0]  # 获取该年的遥感降水量
                # 计算降水量差异并判断是否在允许的范围内
                if (
                    diff_range[0]
                    <= station_rainfall / remote_precipitation
                    <= diff_range[1]
                ):
                    reliable_years.append(True)
                else:
                    reliable_years.append(False)
            else:
                reliable_years.append(None)
                remote_precipitation = None

                # 保存详细的年度数据
            the_result = {
                "STCD": stcd,
                "Latitude": lat,
                "Longitude": lon,
                "Year": year,
                "StationRainfall": station_rainfall,
                "RemotePrecipitation": remote_precipitation,
            }
            results_.append(the_result)

        return reliable_years, results_

    def _station_yearly_sum(self, df_attr, df_station):
        df_station["TM"] = pd.to_datetime(df_station["TM"], errors="coerce")

        # 检查是否有转换失败的日期
        if df_station["TM"].isnull().any():
            self.logger.warning(
                "Warning: Some dates could not be parsed. They will be skipped."
            )
            df_station = df_station.dropna(subset=["TM"])  # 移除无法解析的日期

        df_station["Year"] = df_station["TM"].dt.year
        # 新增筛选汛期数据（6月至10月）
        df_station = df_station[df_station["TM"].dt.month.between(6, 10)]
        df_station["STCD"] = df_station["STCD"].astype(str)  # 将 STCD 转换为字符串
        df_station = df_station.groupby(["STCD", "Year"])["DRP"].sum().reset_index()

        # 合并站点的经纬度和属性表信息
        df_station = pd.merge(
            df_station, df_attr[["STCD", "LGTD", "LTTD"]], on="STCD", how="left"
        )
        # self.logger.debug(df_station)
        return df_station

    def data_check_hourly_extreme(self, basin_id, climate_extreme_value=None, modify = False):
        """
        Check if the daily precipitation values at chosen stations are within a reasonable range.
        Values larger than the climate extreme value are treated as anomalies.
        If no climate_extreme_value is provided, the maximum value in the data is used.

        Parameters
        ----------
        climate_extreme_value : float, optional
            Climate extreme threshold for the region, calculated as 95% of the maximum observed DRP.
            If not provided, will be calculated as 95% of the maximum DRP value in the data.

        Returns
        -------
        df_anomaly_stations_periods : pd.DataFrame
            DataFrame of anomalies with columns: 'STCD', 'TM', 'DRP'.
        """
        trusted_csv_file = os.path.join(self.output_folder, basin_id, "kexin.csv")
        # List of trustworthy station STCDs from the data_check_yearly.
        station_lst = (
            pd.read_csv(trusted_csv_file)["STCD"].drop_duplicates().astype(str).unique()
        )
        # DataFrame containing all daily precipitation data, with columns:
        # 'STCD' (station code), 'TM' (timestamp), and 'DRP' (daily precipitation).
        data_df = self.read_and_concat_csv(basin_id)
        # 如果没有传入气候极值，使用数据中的最大值的 95%
        if climate_extreme_value is None:
            climate_extreme_value = data_df["DRP"].max() * 0.95

        # 过滤出可信站点的数据
        filtered_data = data_df[data_df["STCD"].astype(str).isin(station_lst)]

        # 筛选出超过气候极值的数据
        df_anomaly_stations_periods = filtered_data[
            filtered_data["DRP"] > climate_extreme_value
        ][["STCD", "TM", "DRP"]]
        df_anomaly_stations_periods.to_csv(
            os.path.join(self.output_folder, basin_id, "extreme.csv")
        )
        if modify == True:
            # 遍历文件夹，清除对应行
            folder_path = os.path.join(self.data_folder, basin_id)
            [(
                lambda f: pd.read_csv(f).merge(df_anomaly_stations_periods, on=["STCD", "TM","DRP"], how="left", indicator=True)
                    .query('_merge == "left_only"').drop("_merge", axis=1)
                    .to_csv(f, index=False)
            )(os.path.join(folder_path, file)) for file in os.listdir(folder_path) if file.endswith(".csv")]
 
        return df_anomaly_stations_periods

    def data_check_time_series(
        self,
        basin_id,
        check_type=None,
        gradient_limit=None,
        window_size=None,
        consistent_value=None,
        modify = False
    ):
        """
        Check daily precipitation values at chosen stations for gradient or time consistency anomalies.

        Parameters
        ----------
        basin_id: str
            Basin ID.
        check_type : str
            Type of check to perform: "gradient" for gradient check, "consistency" for time consistency check.
        gradient_limit : float, optional
            Maximum allowable gradient change in precipitation between consecutive days. Used in "gradient" check. Default is 10 mm.
        window_size : int, optional
            Size of the window (in hours) to check for time consistency (used in "consistency" check). Default is 24 hours.
        consistent_value : float, optional
            The specific precipitation value to check for consistency (used in "consistency" check). Default is 0.1 mm.

        Returns
        -------
        pd.DataFrame
            DataFrame of detected anomalies with columns: 'STCD', 'TM', 'DRP', 'Issue' (where applicable).
        """
        # List of trustworthy station STCDs from the data_check_yearly.
        station_lst = (
            pd.read_csv(os.path.join(self.output_folder, basin_id, "kexin.csv"))["STCD"]
            .drop_duplicates()
            .astype(str)
            .unique()
        )
        # DataFrame containing all daily precipitation data, with columns:
        # 'STCD' (station code), 'TM' (timestamp), and 'DRP' (daily precipitation).
        data_df = self.read_and_concat_csv(basin_id)
        # 过滤出可信站点的数据
        filtered_data = data_df[data_df["STCD"].astype(str).isin(station_lst)]
        if check_type == "gradient":
            # 初始化列表来存储所有异常记录
            df_anomalies = self._gradient_limit_check(
                basin_id, filtered_data, gradient_limit
            )

        elif check_type == "consistency":
            # 初始化列表来存储所有异常记录
            df_anomalies = self._consistency_check(
                basin_id, filtered_data, window_size, consistent_value
            )

        else:
            df_anomalies = pd.DataFrame(
                {
                    "STCD": [None],
                    "TM": [None],
                    "DRP": [None],
                    "Issue": [
                        "Invalid check_type. Choose 'gradient' or 'consistency'."
                    ],
                }
            )
        # self.logger.debug(df_anomalies)
        if modify == True:
                # 遍历文件夹，清除对应行
            folder_path = os.path.join(self.data_folder, basin_id)
            [(
                lambda f: pd.read_csv(f).merge(df_anomalies, on=["STCD", "TM","DRP"], how="left", indicator=True)
                    .query('_merge == "left_only"').drop("_merge", axis=1)
                    .to_csv(f, index=False)
            )(os.path.join(folder_path, file)) for file in os.listdir(folder_path) if file.endswith(".csv")]

        return df_anomalies

    def _consistency_check(
        self, basin_id, filtered_data, window_size, consistent_value
    ):
        anomalies = []
        # 使用滑动窗口检测一致性
        for station, station_data in tqdm(filtered_data.groupby("STCD")):
            station_data = station_data.reset_index(drop=True)
            for i in range(len(station_data) - window_size + 1):
                window = station_data.iloc[i : i + window_size]

                # 检查滑动窗口内降雨量是否完全一致且小于指定的阈值
                if window["DRP"].isna().sum() > 0 and (
                    (window["DRP"] < consistent_value).all()
                    and len(window["DRP"].unique()) == 1
                ):
                    anomalies.append(window[["STCD", "TM", "DRP"]])

            # 将所有异常窗口合并成一个 DataFrame
        if anomalies:
            df_anomalies = pd.concat(anomalies).drop_duplicates().reset_index(drop=True)
            df_anomalies["Issue"] = (
                f"Consistent low rain period below {consistent_value} mm"
            )

        else:
            df_anomalies = pd.DataFrame(columns=["STCD", "TM", "DRP", "Issue"])
        df_anomalies.to_csv(
            os.path.join(self.output_folder, basin_id, "consistency.csv")
        )
        return df_anomalies

    def _gradient_limit_check(self, basin_id, filtered_data, gradient_limit):
        anomalies = []

        # 按站点分组并计算双向梯度变化
        for station, station_data in tqdm(filtered_data.groupby("STCD")):
            station_data = station_data.copy()  # 避免修改原始数据
            # 计算前向梯度变化
            station_data["Forward_Change"] = station_data["DRP"].diff()
            # 计算后向梯度变化
            station_data["Backward_Change"] = station_data["DRP"].diff(-1)

            # 筛选出任一方向超过梯度阈值的数据
            station_anomalies = station_data[
                (station_data["Forward_Change"].abs() > gradient_limit)
                | (station_data["Backward_Change"].abs() > gradient_limit)
            ]

            if not station_anomalies.empty:
                anomalies.append(
                    station_anomalies[
                        ["STCD", "TM", "DRP", "Forward_Change", "Backward_Change"]
                    ]
                )

            # 将所有异常记录合并成一个 DataFrame
        if anomalies:
            df_anomalies = pd.concat(anomalies).reset_index(drop=True)
            df_anomalies["Issue"] = "Sudden change in precipitation (forward/backward)"

        else:
            df_anomalies = pd.DataFrame(
                columns=[
                    "STCD",
                    "TM",
                    "DRP",
                    "Forward_Change",
                    "Backward_Change",
                    "Issue",
                ]
            )
        df_anomalies.to_csv(os.path.join(self.output_folder, basin_id, "gradient.csv"))
        return df_anomalies

    def rainfall_clean(self, basin_id, **kwargs):
        """the station gauged rainfall data cleaning pipeline"""
        min_consecutive_years = kwargs.get("min_consecutive_years", 1)
        min_true_percentage = kwargs.get("min_true_percentage", 0.75)
        climate_extreme_value = kwargs.get("climate_extreme_value", 122)
        gradient_limit = kwargs.get("gradient_limit", 120)
        window_size = kwargs.get("window_size", 24)
        modify = kwargs.get("modify", False)
        # 遥感数据初级筛查
        self.data_check_yearly(
            basin_id=basin_id,
            min_true_percentage=min_true_percentage,
            min_consecutive_years=min_consecutive_years,
            modify = modify,
        )
        # 极值监测
        self.data_check_hourly_extreme(
            basin_id=basin_id, climate_extreme_value=climate_extreme_value,
            modify=modify,
        )
        # 时间一致性监测（连续小雨量，梯度）
        self.data_check_time_series(
            basin_id=basin_id,
            check_type="consistency",
            gradient_limit=gradient_limit,
            window_size=window_size,
            consistent_value=0.5,
            modify = modify,
        )
        self.data_check_time_series(
            basin_id=basin_id,
            check_type="gradient",
            gradient_limit=gradient_limit,
            window_size=window_size,
            consistent_value=0.5,
            modify = modify,
        )

class RainfallAnalyzer():
    def __init__(
        self,
        data_folder=None,
        output_folder=None,
        logger_level=logging.INFO,
    ):
        # 初始化 logger
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logger_level)
        if not self.logger.hasHandlers():
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(handler)

        self.data_folder = data_folder
        self.output_folder = output_folder
        self.logger_level = logger_level
        self.data_source_description = self.set_data_source_describe()
        self._check_file_format()
        # self.station_info = self.read_site_info()

    def set_data_source_describe(self):
        data_source_dir = self.data_folder
        output_folder = self.output_folder
        output_plot = os.path.join(output_folder, "plot")
        if not os.path.exists(output_plot):
            os.makedirs(output_plot)
        output_log = os.path.join(
            output_plot,
            "summary_log.txt",
        )
        stations_csv_path = os.path.join(data_source_dir, "stations")
        # 站点表，其中ID列带有前缀‘pp_’
        shp_folder = os.path.join(data_source_dir, "basins_shp")
        return collections.OrderedDict(
            STATIONS_CSV_PATH=stations_csv_path,
            SHP_FOLDER=shp_folder,
            OUTPUT_LOG=output_log,
            OUTPUT_PLOT=output_plot,
        )

    def _check_file_format(self):
        # check if the file format is correct
        pass

    def filter_and_save_csv(self, basin_id):
        """
        TODO: need use RainfallCleaner to filter the data
        筛选降雨数据，根据每年的总降雨量（DRP）进行过滤，保留符合最低和最高阈值的数据。

        参数：
        input_folder - 包含降雨数据的文件夹路径。
        lower_bound - 降雨量最低阈值。
        upper_bound - 降雨量最高阈值。

        返回：
        过滤后的降雨数据DataFrame。
        """
        self.logger.info("Filtering data by yearly total DRP")
        input_folder = os.path.join(self.data_folder, basin_id)
        filtered_data_list = []
        for file in os.listdir(input_folder):
            if file.endswith(".csv"):
                file_path = os.path.join(input_folder, file)
                data = pd.read_csv(file_path)
                data["TM"] = pd.to_datetime(data["TM"], errors="coerce")
                data["TM"] = pd.to_datetime(
                    data["TM"], format="%Y-%m-%d %H:%M:%S.%f", errors="coerce"
                )
                data["DRP"] = data["DRP"].astype(float)

                data["ID"] = file.replace(".csv", "")
                for year, group in data.groupby(data["TM"].dt.year):
                    drp_sum = group["DRP"].sum()
                    filtered_data_list.append(group)
        if filtered_data_list:
            return pd.concat(filtered_data_list, ignore_index=True)
        else:
            return pd.DataFrame()

    def read_data(self, basin_id):
        """
        读取站点信息和流域shapefile数据。

        Parameters
        ----------
        basin_id : str
            basin ID

        Returns
        -------
        stations_df: pd.DataFrame
            station information DataFrame
        basin : geopandas.GeoDataFrame
            basin shapefile GeoDataFrame
        """
        # station info CSV file
        stations_csv_path = os.path.join(
            self.data_source_description["STATIONS_CSV_PATH"],
            f"stations.csv",
        )
        stations_df = pd.read_csv(stations_csv_path)
        stations_df = stations_df.dropna(subset=["LON", "LAT"])
        # basin shapefile path
        basin_shp_path = os.path.join(
            self.data_source_description["SHP_FOLDER"], basin_id, f"{basin_id}.shp"
        )
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
        self.logger.info("Processing stations within the basin")
        gdf_stations = gpd.GeoDataFrame(
            stations_df,
            geometry=[Point(xy) for xy in zip(stations_df.LON, stations_df.LAT)],
            crs="EPSG:4326",
        )
        gdf_stations = gdf_stations.to_crs(basin.crs)
        stations_within_basin = sjoin(gdf_stations, basin, predicate="within")
        self.logger.info(
            f"Found {len(stations_within_basin)} stations within the basin"
        )
        self.logger.debug(stations_within_basin)
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
        self.logger.debug(f"Basin area: {basin_area}")

        # 计算原始泰森多边形的总面积
        total_original_area = gdf_polygons["original_area"].sum()
        self.logger.debug(
            f"Total original Voronoi polygons area: {total_original_area}"
        )

        # 将多边形裁剪到流域边界
        clipped_polygons = gpd.clip(gdf_polygons, basin)
        clipped_polygons["clipped_area"] = clipped_polygons.geometry.area
        clipped_polygons["area_ratio"] = (
            clipped_polygons["clipped_area"] / clipped_polygons["clipped_area"].sum()
        )

        # 计算裁剪后泰森多边形的总面积
        total_clipped_area = clipped_polygons["clipped_area"].sum()
        self.logger.debug(f"Total clipped Voronoi polygons area: {total_clipped_area}")

        # 打印年度数据汇总并将其追加到日志文件中
        log_entries = [
            f"Basin area: {basin_area}",
            f"Total original Voronoi polygons area: {total_original_area}",
            f"Total clipped Voronoi polygons area: {total_clipped_area}",
        ]
        for entry in log_entries:
            self.logger.info(entry + "\n")
            continue

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

        return merged_data.groupby("TM")["weighted_rainfall"].sum().reset_index()

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
        self.logger.debug(f"Displaying results for year {year}")

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
        file_path = f"{self.data_source_description['OUTPUT_PLOT']}/{file_name}"
        # 保存图像
        plt.savefig(file_path)
        # plt.show()

        # 输出站点名称和数量
        station_names = valid_stations["ID"].tolist()
        station_count = len(station_names)
        self.logger.debug(f"Stations for year {year}: {station_names}")
        self.logger.debug(f"Total number of stations: {station_count}")

        # 输出对应年份的数据
        filtered_yearly_data = yearly_data[yearly_data["ID"].isin(station_names)]

        yearly_summary = (
            filtered_yearly_data.groupby("ID")
            .agg({"STCD": "first", "DRP": "sum"})
            .reset_index()
        )
        self.logger.debug(f"Yearly data summary for year {year}:\n{yearly_summary}")

        # 输出平均雨量数据
        mean_rainfall = average_rainfall["mean_rainfall"].sum()
        self.logger.debug(f"Average rainfall for year {year}: {mean_rainfall}")

        # 追加日志
        # 打印年度数据汇总并将其追加到日志文件中
        log_entries = [
            f"BASINS: {basin['BASIN_ID'].iloc[0]}",
            f"Displaying results for year {year}",
            f"Stations for year {year}: {station_names}",
            f"Total number of stations: {station_count}",
            f"Yearly data summary for year {year}:\n{yearly_summary}",
            f"Average rainfall for year {year}: {mean_rainfall}\n",
        ]
        for entry in log_entries:
            self.logger.info(entry + "\n")
            

    def process_basin(self, basin_id, filtered_data):
        """
        处理每个流域的降雨数据，计算泰森多边形和面平均降雨量。

        参数：
        filtered_data - 预先过滤的降雨数据DataFrame。
        output_folder - 输出文件夹路径。
        """
        all_years_rainfall = []
        stations_df, basin = self.read_data(basin_id)

        years = filtered_data["TM"].dt.year.unique()

        for year in sorted(years):
            self.logger.info(f"Processing basin {basin_id} for year {year}")
            # 打印年度数据汇总并将其追加到日志文件中
            log_file = self.data_source_description["OUTPUT_LOG"]
            with open(log_file, "a") as f:
                f.write(f"Processing basin {basin_id} for year {year}\n")
            yearly_data = filtered_data[filtered_data["TM"].dt.year == year]

            if yearly_data.empty:
                # 打印年度数据汇总并将其追加到日志文件中
                self.logger.info(f"No valid data for basin {basin_id} in year {year}")
                

            # 筛选符合条件的每年站点数据
            yearly_stations = yearly_data["ID"].unique()
            self.logger.debug(yearly_stations)
            valid_stations = self.process_stations(stations_df, basin)
            self.logger.debug(valid_stations["ID"])
            valid_stations = valid_stations[valid_stations["ID"].isin(yearly_stations)]
            self.logger.debug("11111111111111111111111111")
            self.logger.debug(valid_stations.head())

            if valid_stations.empty:
                # 打印年度数据汇总并将其追加到日志文件中
                self.logger.info(
                    f"No valid stations for basin {basin_id} in year {year}\n"
                )
                

            thiesen_polygons_year = self.calculate_voronoi_polygons(
                valid_stations, basin
            )
            average_rainfall = self.calculate_weighted_rainfall(
                thiesen_polygons_year, yearly_data
            )
            average_rainfall.columns = ["TM", "mean_rainfall"]
            basin_id = os.path.splitext(basin_id)[0]
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
            self._concat_yearly_data(all_years_rainfall, basin_id)
        else:
            self.logger.info(f"No valid data for basin {basin_id}")
            

    def _concat_yearly_data(self, all_years_rainfall, basin_id):
        final_result = pd.concat(all_years_rainfall, ignore_index=True)
        basin_output_folder = os.path.join(self.output_folder, basin_id)
        os.makedirs(basin_output_folder, exist_ok=True)
        output_file = os.path.join(basin_output_folder, f"{basin_id}_rainfall.csv")
        final_result.to_csv(output_file, index=False)
        self.logger.info(f"Result for basin {basin_id} saved to {output_file}")

    def basins_polygon_mean(self, basin_ids):
        """
        basin mean rainfall calculation pipeline

        Parameters
        ----------
        basin_ids : list
            Basin ID list
        shp_folder - 流域shapefile文件夹路径。
        rainfall_data_folder - 降雨数据文件夹路径。
        lower_bound - 降雨量最低阈值。
        upper_bound - 降雨量最高阈值。
        output_folder - 输出文件夹路径。
        """
        # 先筛选降雨数据，保留符合最低阈值的数据
        for basin_id in basin_ids:
            filtered_data = self.filter_and_save_csv(basin_id)
            self.process_basin(basin_id, filtered_data)
