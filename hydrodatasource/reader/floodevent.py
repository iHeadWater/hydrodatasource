"""
Author: Wenyu Ouyang
Date: 2025-01-19 18:05:00
LastEditTime: 2025-08-17 10:09:14
LastEditors: Wenyu Ouyang
Description: 流域场次数据处理类 - 继承自SelfMadeHydroDataset
FilePath: \hydromodeld:\Code\hydrodatasource\hydrodatasource\reader\floodevent.py
Copyright (c) 2023-2026 Wenyu Ouyang. All rights reserved.
"""

import glob
import re
import pandas as pd
import numpy as np
import os
import xarray as xr
from datetime import datetime, timedelta
from typing import Any, List, Dict, Optional, Tuple, Union
from hydroutils import hydro_event
from hydrodatasource.utils.utils import streamflow_unit_conv
from hydrodatasource.reader.data_source import SelfMadeHydroDataset
from hydrodatasource.configs.config import CACHE_DIR


class FloodEventDatasource(SelfMadeHydroDataset):
    """
    Flood event dataset processing class

    Inherits from SelfMadeHydroDataset, specifically designed for
    processing individual flood event data, including event extraction functions.
    """

    def __init__(
        self,
        data_path: str,
        dataset_name: str = "songliaorrevents",
        time_unit: Optional[List[str]] = None,
        rain_key: str = "rain",
        pet_key: str = "ES",
        net_rain_key: str = "net_rain",
        obs_flow_key: str = "inflow",
        warmup_length: int = 0,
        **kwargs,
    ):
        """
        Initialize the flood event dataset.

        Parameters
        ----------
        data_path : str
            Path to the data.
        dataset_name : str, optional
            Name of the dataset.
        time_unit : list of str, optional
            List of time units, default is ["3h"].
        rain_key : str, optional
            Key name for rain data, default is "rain".
        net_rain_key : str, optional
            Key name for net rain data, default is "net_rain".
        obs_flow_key : str, optional
            Key name for observed flow data, default is "inflow".
        warmup_length : int, optional
            Number of time steps to include before flood event starts as warmup
            period, default is 0.
        **kwargs
            Additional keyword arguments passed to the parent class.
        """
        if time_unit is None:
            time_unit = ["3h"]

        # Derive delta_t_hours from time_unit
        # 从time_unit推导出时间步长（小时）
        primary_time_unit = time_unit[0]  # 使用第一个时间单位作为主要单位
        delta_t_hours = FloodEventDatasource._parse_time_unit_to_hours(
            primary_time_unit
        )

        # Store constants as instance attributes
        self.rain_key = rain_key
        self.pet_key = pet_key  # 假设数据集中蒸散发的列名为 'ES'
        self.net_rain_key = net_rain_key
        self.obs_flow_key = obs_flow_key
        self.warmup_length = warmup_length
        self.delta_t_hours = delta_t_hours
        self.delta_t_seconds = self.delta_t_hours * 3600.0

        super().__init__(
            data_path=data_path,
            download=False,
            time_unit=time_unit,
            dataset_name=dataset_name,
            **kwargs,
        )

        # 验证数据目录是否存在
        if not os.path.exists(self.data_source_dir):
            raise FileNotFoundError(
                f"❌ 数据集目录不存在: {self.data_source_dir}\n"
                f"\n配置信息:"
                f"\n  data_path     = {data_path}"
                f"\n  dataset_name  = {dataset_name}"
                f"\n  拼接后路径    = {self.data_source_dir}"
                f"\n\n请检查:"
                f"\n  1. data_path 是否指向数据集的父目录"
                f"\n  2. dataset_name 拼写是否正确"
                f"\n  3. 数据集目录是否实际存在"
            )

        # 验证必需的子目录
        required_subdirs = ["attributes", "timeseries"]
        missing_dirs = []
        for subdir in required_subdirs:
            subdir_path = os.path.join(self.data_source_dir, subdir)
            if not os.path.exists(subdir_path):
                missing_dirs.append(subdir)

        if missing_dirs:
            raise FileNotFoundError(
                f"❌ 数据集目录结构不完整: {self.data_source_dir}\n"
                f"缺少以下子目录: {', '.join(missing_dirs)}"
            )

    def get_constants(self):
        """
        Get the constant values used by this datasource.

        Returns
        -------
        dict
            Dictionary containing constant values with keys:
            - 'net_rain_key': Key name for net rain data
            - 'obs_flow_key': Key name for observed flow data
            - 'delta_t_hours': Time step in hours (derived from time_unit)
            - 'delta_t_seconds': Time step in seconds
            - 'warmup_length': Number of warmup time steps
        """
        return {
            "net_rain_key": self.net_rain_key,
            "obs_flow_key": self.obs_flow_key,
            "delta_t_hours": self.delta_t_hours,
            "delta_t_seconds": self.delta_t_seconds,
            "warmup_length": self.warmup_length,
        }

    @staticmethod
    def _parse_time_unit_to_hours(time_unit_str: str) -> float:
        """
        Parse time unit string to hours.

        Parameters
        ----------
        time_unit_str : str
            Time unit string like "1h", "3h", "1D", "2D", "8D", etc.

        Returns
        -------
        float
            Time step in hours.

        Examples
        --------
        >>> FloodEventDatasource._parse_time_unit_to_hours("3h")
        3.0
        >>> FloodEventDatasource._parse_time_unit_to_hours("1D")
        24.0
        >>> FloodEventDatasource._parse_time_unit_to_hours("2D")
        48.0
        """
        import re

        # 使用正则表达式解析时间单位
        pattern = r"^(\d+)([hdHD])$"
        match = re.match(pattern, time_unit_str.strip())

        if not match:
            raise ValueError(
                f"Invalid time_unit format: '{time_unit_str}'. "
                f"Expected format: <number><unit> where unit is 'h' or 'D' "
                f"(e.g., '1h', '3h', '1D', '2D')"
            )

        number = int(match.group(1))
        unit = match.group(2).upper()

        if unit == "H":
            return float(number)
        elif unit == "D":
            return float(number * 24)
        else:
            # 这里实际上不会执行到，因为正则表达式已经限制了单位
            raise ValueError(f"Unsupported time unit: {unit}")

    def extract_flood_events(
        self, df: pd.DataFrame, include_peak_obs: bool = True
    ) -> List[Dict]:
        """
        提取洪水事件并转换为标准格式字典

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing station data.
        include_peak_obs : bool, default=True
            是否包含洪峰观测值

        Returns
        -------
        List[Dict]
            标准格式的事件字典列表，与现有算法完全兼容
        """
        # 调用hydroutils提取事件
        flood_events = hydro_event.extract_flood_events(
            df=df,
            warmup_length=self.warmup_length,
            flood_event_col="flood_event",
            time_col="time",
        )

        all_events = []
        for event_idx, event in enumerate(flood_events):
            event_data = event["data"]

            # 提取各列数据
            rain = event_data[self.rain_key].values
            ES = event_data[self.pet_key].values
            inflow = event_data[self.obs_flow_key].values
            flood_event_markers = event_data["flood_event"].values

            # 提取时间信息
            if "time" in event_data.columns:
                time_array = event_data["time"].values
            else:
                time_array = event_data.index.values

            # 创建标准格式字典
            event_dict = self._create_event_dict(
                rain=rain,
                inflow=inflow,
                event_name=event["event_name"],
                ES=ES,
                include_peak_obs=include_peak_obs,
                flood_event_markers=flood_event_markers,
            )

            if event_dict is not None:
                # 添加时间信息和事件ID到事件字典
                event_dict["time"] = time_array
                event_dict["event_id"] = event_idx + 1  # 1-based event ID
                event_dict["event_name"] = event["event_name"]
                all_events.append(event_dict)

        return all_events

    def _create_event_dict(
        self,
        rain: np.ndarray,
        inflow: np.ndarray,
        event_name: str,
        ES: np.ndarray,
        include_peak_obs: bool = True,
        flood_event_markers: Optional[np.ndarray] = None,
    ) -> Optional[Dict]:
        """
        Transform rain, net_rain and inflow arrays into a standard event dictionary format

        Parameters
        ----------
        rain: np.ndarray
            rain array
        inflow: np.ndarray
            inflow array
        event_name: str
            洪峰日期（8位数字格式）
        ES: np.ndarray
            蒸散发数组
        include_peak_obs: bool
            是否包含洪峰观测值
        flood_event_markers: np.ndarray, optional
            flood_event markers indicating which time steps belong to actual
            flood event (>0) vs warmup period (0)

        Returns
        -------
            Dict: 标准格式的事件字典，与uh_utils.py完全兼容，
            包含flood_event_markers信息
        """
        try:
            # 计算有效降雨时段数
            valid_rain_mask = ~np.isnan(rain) & (rain > 0)
            m_eff = np.sum(valid_rain_mask)

            if m_eff == 0:
                return None

            # 验证径流数据
            if np.nansum(inflow) < 1e-6:
                return None

            # 创建标准格式字典（与uh_utils.py期望的key完全一致）
            event_dict = {
                self.rain_key: rain,
                self.obs_flow_key: inflow,  # 观测径流
                "ES": ES,
                "m_eff": m_eff,  # 净雨时段数
                "n_specific": len(rain),  # 单位线长度
                "filepath": f"event_{event_name}.csv",  # 避免KeyError
            }

            # 添加洪水事件标记（如果提供）
            if flood_event_markers is not None:
                event_dict["flood_event_markers"] = flood_event_markers

            # 添加洪峰观测值
            if include_peak_obs:
                peak_flow = np.nanmax(inflow)
                if peak_flow < 1e-6:
                    return {}
                event_dict["peak_obs"] = peak_flow

            return event_dict

        except Exception:
            return {}

    def load_1basin_flood_events(
        self,
        station_id: Optional[str] = None,
        flow_unit: str = "mm/3h",
        include_peak_obs: bool = True,
        verbose: bool = True,
        **kwargs,
    ) -> Optional[List[Dict]]:
        """
        加载洪水事件数据

        Parameters
        ----------
        station_id:
            指定站点ID，如果为None则处理所有站点
        flow_unit
            Unit of streamflow, default is "mm/3h".
        include_peak_obs:
            是否包含洪峰观测值
        verbose:
            是否打印详细信息

        Returns
        -------
            List[Dict]: 标准格式的事件字典列表，与现有算法完全兼容
        """
        # 获取流域面积
        basin_area_km2 = None

        if station_id:
            basin_area_km2 = self.read_area([station_id])
        else:
            basin_area_km2 = None

        try:
            if verbose:
                print("🔄 Loading floodevent data...")
                if station_id:
                    print(f"   station_id: {station_id}")

            xr_ds = self.read_ts_xrdataset(
                gage_id_lst=[station_id],
                t_range=["1960-01-01", "2024-12-31"],
                var_lst=["rain", "inflow", "flood_event", "ES"],
                recache=False,  # 是否强制重新缓存，确保数据最新
                **kwargs,
            )[self.time_unit[0]]

            xr_ds["inflow"] = streamflow_unit_conv(
                xr_ds[["inflow"]],
                target_unit=flow_unit,
                area=basin_area_km2,
            )["inflow"]
            df = xr_ds.to_dataframe()
            if df is None:
                return None

            # 直接提取洪水事件并转换为标准格式
            all_events = self.extract_flood_events(
                df.loc[station_id].reset_index(), include_peak_obs
            )

            if not all_events:
                if verbose:
                    print(f"  ⚠️  {station_id}: No valid flood events were found")
                return None

            if verbose:
                print(
                    f"  ✅ {station_id}: Successfully processed {len(all_events)} flood events"
                )
                print(f"✅ Successfully load {len(all_events)} flood events")

            return all_events

        except Exception as e:
            if verbose:
                print(f"❌ An error occurred while loading floodevent data: {str(e)}")
            return None

    def parse_augmented_file_metadata(self, augmented_file_path: str) -> Dict:
        """
        解析增强文件的元信息

        Parameters
        ----------
        augmented_file_path : str
            增强文件的路径

        Returns
        -------
        Dict
            包含源场次信息的字典，包括起始时间、结束时间、源文件名等
        """
        metadata = {}

        with open(augmented_file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("#"):
                    if "Source:" in line:
                        source_file = line.split("Source:")[1].strip()
                        metadata["source_file"] = source_file
                        # 从源文件名提取起始时间
                        if "event_" in source_file and ".csv" in source_file:
                            time_part = source_file.replace("event_", "").replace(
                                ".csv", ""
                            )
                            if "_" in time_part:
                                start_time_str, end_time_str = time_part.split("_")
                                metadata["original_start_time"] = start_time_str
                                metadata["original_end_time"] = end_time_str
                    elif "Start Time:" in line:
                        metadata["augmented_start_time"] = line.split("Start Time:")[
                            1
                        ].strip()
                    elif "End Time:" in line:
                        metadata["augmented_end_time"] = line.split("End Time:")[
                            1
                        ].strip()
                    elif "Scale Factor:" in line:
                        metadata["scale_factor"] = float(
                            line.split("Scale Factor:")[1].strip()
                        )
                    elif "Sample ID:" in line:
                        metadata["sample_id"] = int(line.split("Sample ID:")[1].strip())
                    elif "Score:" in line:
                        # 提取评分信息
                        import re
                        score_match = re.search(r"#\s*Score:\s*(\d+)", line)
                        if score_match:
                            score = int(score_match.group(1))
                            # 限制评分范围0-100
                            metadata["Score"] = max(0, min(100, score))
                else:
                    break

        # 处理缺失Score的情况：根据时间分界线判断
        if "Score" not in metadata:
            if "augmented_start_time" in metadata:
                start_year = int(metadata["augmented_start_time"][:4])
                if start_year <= 2025:
                    # 真实数据：默认100分
                    metadata["Score"] = 100
                else:
                    # 2026+增强数据无评分：标记为None（后续跳过）
                    metadata["Score"] = None
            else:
                # 无法判断时间：默认100分
                metadata["Score"] = 100

        return metadata

    def get_warmup_period_data(
        self,
        original_start_time: str,
        original_end_time: str,
        station_id: str,
        warmup_hours: int = 240,
    ) -> Optional[pd.DataFrame]:
        """
        获取原始场次前面的预热期数据

        Parameters
        ----------
        original_start_time : str
            原始场次起始时间 (YYYYMMDDHH格式)
        original_end_time : str
            原始场次结束时间 (YYYYMMDDHH格式)
        station_id : str
            站点ID
        warmup_hours : int, optional
            预热期小时数，默认240小时(10天)

        Returns
        -------
        Optional[pd.DataFrame]
            预热期数据，包含time, rain, gen_discharge, obs_discharge列
        """
        try:
            # 使用字符串操作处理时间
            year = original_start_time[:4]
            month = original_start_time[4:6]
            day = original_start_time[6:8]
            hour = original_start_time[8:10]

            # 构造时间字符串
            start_time = f"{year}-{month}-{day} {hour}:00:00"

            # 由于时间超出pandas范围，我们暂时使用一个基准年份进行计算
            base_year = "2000"
            base_start = f"{base_year}-{month}-{day} {hour}:00:00"
            base_start_dt = datetime.strptime(base_start, "%Y-%m-%d %H:%M:%S")

            # 计算预热期时间
            base_warmup_start = base_start_dt - timedelta(hours=warmup_hours)
            base_warmup_end = base_start_dt - timedelta(hours=self.delta_t_hours)

            # 替换回原始年份
            warmup_start = base_warmup_start.strftime(f"{year}-%m-%d %H:%M:%S")
            warmup_end = base_warmup_end.strftime(f"{year}-%m-%d %H:%M:%S")

            # 读取预热期数据
            xr_ds = self.read_ts_xrdataset(
                gage_id_lst=[station_id],
                t_range=[warmup_start, warmup_end],
                var_lst=["rain", "inflow", "flood_event", "ES"],
            )["3h"]

            if xr_ds is None:
                return None

            # 转换为DataFrame
            df = xr_ds.to_dataframe().reset_index()
            df = df[df["basin"] == station_id].copy()

            # 重命名列以匹配增强文件格式
            df = df.rename(columns={"inflow": "obs_discharge"})
            df["gen_discharge"] = df["obs_discharge"]

            return df[["time", "rain", "gen_discharge", "obs_discharge"]]
        except Exception as e:
            print(f"Failed to warmup data: {e}")
            return None

    def adjust_warmup_time_to_augmented_year(
        self, warmup_df: pd.DataFrame, augmented_start_time: str
    ) -> pd.DataFrame:
        """
        调整预热期数据的年份到增强数据的年份

        Parameters
        ----------
        warmup_df : pd.DataFrame
            预热期数据
        augmented_start_time : str
            增强数据的起始时间 (YYYYMMDDHH格式)

        Returns
        -------
        pd.DataFrame
            调整年份后的预热期数据
        """
        df = warmup_df.copy()

        # 获取增强数据的年份
        aug_year = augmented_start_time

        # 调整时间列的年份（字符串操作）
        df["time"] = df["time"].astype(str)
        df["time"] = df["time"].apply(lambda x: aug_year + x[4:])

        return df

    def concatenate_warmup_and_augmented_data(
        self,
        warmup_df: pd.DataFrame,
        augmented_file_path: str,
        use_score_as_weight: bool = False,  # 新增参数
        score_threshold: int = 60,  # 新增参数
    ) -> pd.DataFrame:
        """
        拼接预热期数据和增强场次数据

        Parameters
        ----------
        warmup_df : pd.DataFrame
            预热期数据（仅在增强CSV未包含预热期时使用）
        augmented_file_path : str
            增强文件路径

        Returns
        -------
        pd.DataFrame
            拼接后的完整数据
        """
        # 读取增强数据
        aug_df = pd.read_csv(augmented_file_path, comment="#")
        # 将时间列保持为字符串格式，避免超出pandas时间戳范围限制
        aug_df["time"] = aug_df["time"].astype(str)

        # 解析元数据获取event_id
        metadata = self.parse_augmented_file_metadata(augmented_file_path)

        # 检查增强CSV是否已包含flood_event列（表示已包含预热期）
        if "flood_event" in aug_df.columns:
            # CSV文件已包含预热期和洪水期数据，无需拼接外部warmup

            # 解析元数据获取评分
            score = metadata.get("Score", None)

            # 根据模式处理 flood_event 列
            if use_score_as_weight and score is not None:
                # 评分加权模式：将评分存储到 flood_event 列
                score = int(score)

                def apply_score_weighting(flood_event_value):
                    if flood_event_value == 0:
                        # 非洪水保持为 0
                        return 0
                    elif flood_event_value == 1:
                        # 洪水事件：根据评分决定
                        if score < score_threshold:
                            return 1  # 低质量：标记为 1
                        else:
                            return score  # 高质量：使用实际评分
                    else:
                        return flood_event_value  # 其他值保持不变

                aug_df["flood_event"] = aug_df["flood_event"].apply(apply_score_weighting)
                print(f"      📊 Applied score weighting: score={score}, threshold={score_threshold}")

            # 添加 event_id 列
            if "augmented_start_time" in metadata and "augmented_end_time" in metadata:
                event_id = f"event_{metadata['augmented_start_time']}_{metadata['augmented_end_time']}"
                aug_df["event_id"] = aug_df["flood_event"].apply(
                    lambda x: event_id if x >= 1 else ""  # 修改：>= 1 包含所有洪水事件
                )
            else:
                aug_df["event_id"] = ""

            return aug_df

        # 旧版CSV文件不包含flood_event列，需要拼接外部warmup
        # 获取增强数据的起始时间，用于调整预热期数据的年份
        if not aug_df.empty:
            # 从字符串格式的时间中提取年份
            aug_start_time = aug_df["time"].min()[:4]
            # 调整预热期数据的年份到增强数据的年份
            adjusted_warmup_df = self.adjust_warmup_time_to_augmented_year(
                warmup_df, aug_start_time
            )
        else:
            adjusted_warmup_df = warmup_df

        # 确保所有时间列都是字符串格式
        adjusted_warmup_df["time"] = adjusted_warmup_df["time"].astype(str)

        # 为预热期数据和增强数据添加标记列
        adjusted_warmup_df["flood_event"] = 0  # 预热期数据标记为0

        # 根据模式设置洪水期的 flood_event 值
        score = metadata.get("Score", None)

        if use_score_as_weight and score is not None:
            # 评分加权模式
            score = int(score)
            if score < score_threshold:
                aug_df["flood_event"] = 1  # 低质量：标记为 1
            else:
                aug_df["flood_event"] = score  # 高质量：使用实际评分
            print(f"      📊 Applied score weighting: score={score}, threshold={score_threshold}")
        else:
            # 阈值过滤模式
            aug_df["flood_event"] = 1  # 洪水期数据统一标记为1

        # 添加event_id列（用于后续评分匹配）
        if "augmented_start_time" in metadata and "augmented_end_time" in metadata:
            event_id = f"event_{metadata['augmented_start_time']}_{metadata['augmented_end_time']}"
            aug_df["event_id"] = event_id
        else:
            aug_df["event_id"] = ""
        adjusted_warmup_df["event_id"] = ""  # 预热期无event_id

        # 拼接数据并按字符串格式的时间排序
        combined_df = pd.concat([adjusted_warmup_df, aug_df], ignore_index=True)
        # 使用字符串比较进行排序
        combined_df = combined_df.sort_values(
            "time", key=lambda x: x.astype(str)
        ).reset_index(drop=True)

        return combined_df

    def rename_dataframe_columns(
        self, df: pd.DataFrame, custom_mapping: dict = None
    ) -> pd.DataFrame:
        """
        重命名数据框的列名，包括默认的映射和自定义映射

        Parameters
        ----------
        df : pd.DataFrame
            需要重命名列的数据框
        custom_mapping : dict, optional
            自定义的列名映射字典，例如 {'old_name': 'new_name'}

        Returns
        -------
        pd.DataFrame
            列名重命名后的数据框
        """
        # 默认的列名映射
        default_mapping = {
            "gen_discharge": "inflow",
            # 在这里添加其他默认的列名映射
        }

        # 如果提供了自定义映射，则更新默认映射
        if custom_mapping:
            default_mapping.update(custom_mapping)

        # 获取数据框中实际存在的列
        existing_columns = set(df.columns)

        # 只重命名实际存在的列
        mapping_to_apply = {
            old: new for old, new in default_mapping.items() if old in existing_columns
        }

        # 如果有需要重命名的列，则进行重命名
        if mapping_to_apply:
            df = df.rename(columns=mapping_to_apply)
            renamed_cols = ", ".join(
                [f"{old}->{new}" for old, new in mapping_to_apply.items()]
            )

        return df

    def create_xarray_dataset_from_augdf(
        self, df: pd.DataFrame, station_id: str, time_unit: str = "3h"
    ) -> xr.Dataset:
        """
        将时间序列DataFrame转换为xarray Dataset格式

        Parameters
        ----------
        df : pd.DataFrame
            时间序列数据
        station_id : str
            站点ID
        time_unit : str, optional
            时间单位，默认"3h"

        Returns
        -------
        xr.Dataset
            xarray格式的数据集
        """
        # The gen_discharge is the generated discharge by the data augmentation method
        # 创建数据集字典
        # 注意：维度顺序必须为 ["basin", "time"] 以与hydrodatasource标准格式兼容
        data_vars = {}

        # 添加降雨数据
        if "rain" in df.columns:
            data_vars["rain"] = (
                ["basin", "time"],  # 修正：维度顺序改为 basin 在前
                df[["rain"]].values.reshape(1, -1),  # 修正：形状改为 (1, n_time)
            )

        # 添加生成的流量数据
        if "inflow" in df.columns:
            data_vars["inflow"] = (
                ["basin", "time"],
                df[["inflow"]].values.reshape(1, -1),
            )

        if "gen_discharge" in df.columns:
            data_vars["inflow"] = (
                ["basin", "time"],
                df[["gen_discharge"]].values.reshape(1, -1),
            )

        # 添加观测流量数据
        # if "obs_discharge" in df.columns:
        #     data_vars["obs_discharge"] = (
        #         ["basin", "time"],
        #         df[["obs_discharge"]].values.reshape(1, -1),
        #     )

        # 洪水期标记
        if "flood_event" in df.columns:
            data_vars["flood_event"] = (
                ["basin", "time"],
                df[["flood_event"]].values.reshape(1, -1),
            )

        # 添加蒸散发数据
        if "ES" in df.columns:
            data_vars["ES"] = (
                ["basin", "time"],
                df[["ES"]].values.reshape(1, -1),
            )

        # 创建数据集
        ds = xr.Dataset(
            data_vars,
            coords={"time": df["time"].values, "basin": [station_id]},
        )

        # 添加数据集属性
        ds.attrs["description"] = "Augmented hydrological time series data"
        ds.attrs["station_id"] = station_id
        ds.attrs["time_unit"] = time_unit
        ds.attrs["creation_time"] = datetime.now().isoformat()

        # 为每个变量添加 units 属性
        for var_name in ds.data_vars:
            if var_name == "rain":
                ds[var_name].attrs["units"] = f"mm/{time_unit}"  # 降雨单位
            elif var_name in ["inflow", "gen_discharge", "obs_discharge"]:
                ds[var_name].attrs["units"] = "m^3/s"  # 流量单位（包括生成的和观测的）
            elif var_name == "flood_event":
                ds[var_name].attrs["units"] = "dimensionless"  # 无量纲
            elif var_name == "ES":
                ds[var_name].attrs["units"] = f"mm/{time_unit}"  # 蒸散发单位
            else:
                ds[var_name].attrs["units"] = "unknown"  # 默认值

            # 添加变量描述
            if var_name == "rain":
                ds[var_name].attrs["description"] = "降雨量"
            elif var_name == "inflow":
                ds[var_name].attrs["description"] = "生成的流量"
            elif var_name == "gen_discharge":
                ds[var_name].attrs["description"] = "生成的流量"
            elif var_name == "obs_discharge":
                ds[var_name].attrs["description"] = "观测流量"
            elif var_name == "flood_event":
                ds[var_name].attrs["description"] = "洪水事件标记"
            elif var_name == "ES":
                ds[var_name].attrs["description"] = "蒸散发"

        return ds

    def save_augmented_timeseries_to_cache(
        self, ds: xr.Dataset, station_ids: Union[str, List[str]], time_unit: str = "3h"
    ) -> str:
        """
        将增强时间序列数据保存到cache目录

        Parameters
        ----------
        ds : xr.Dataset
            要保存的数据集
        station_ids : Union[str, List[str]]
            站点ID或站点ID列表
        time_unit : str, optional
            时间单位，默认"3h"

        Returns
        -------
        str
            保存的文件路径
        """
        # 兼容单个站点的情况
        if isinstance(station_ids, str):
            station_ids = [station_ids]

        # 构造文件名，参考原有的命名规则，加上dataaugment前缀
        prefix = f"{self.dataset_name}_dataaugment_"
        first_station = station_ids[0]
        last_station = station_ids[-1]
        cache_file_name = (
            f"{prefix}timeseries_{time_unit}_batch_{first_station}_{last_station}.nc"
        )
        cache_file_path = os.path.join(CACHE_DIR, cache_file_name)

        # 保存数据
        ds.to_netcdf(cache_file_path)

        return cache_file_path

    def read_ts_xrdataset_augmented(
        self,
        gage_id_lst: Optional[List[str]] = None,
        t_range: Optional[List[str]] = None,
        var_lst: Optional[List[str]] = None,
        time_unit: str = "3h",
        **kwargs,
    ) -> Dict:
        """
        读取增强的时间序列数据，优先从dataaugment缓存文件读取

        Parameters
        ----------
        gage_id_lst : Optional[List[str]], optional
            站点ID列表
        t_range : Optional[List[str]], optional
            时间范围
        var_lst : Optional[List[str]], optional
            变量列表
        time_unit : str, optional
            时间单位，默认"3h"
        **kwargs
            其他参数

        Returns
        -------
        Dict
            包含增强数据的字典，格式与read_ts_xrdataset一致
        """
        if gage_id_lst is None or len(gage_id_lst) == 0:
            return self.read_ts_xrdataset(gage_id_lst, t_range, var_lst, **kwargs)

        # 构造增强数据缓存文件路径，支持多流域
        prefix = f"{self.dataset_name}_dataaugment_"
        first_station = gage_id_lst[0]
        last_station = gage_id_lst[-1]
        cache_file_name = (
            f"{prefix}timeseries_{time_unit}_batch_{first_station}_{last_station}.nc"
        )
        cache_file_path = os.path.join(CACHE_DIR, cache_file_name)

        # 检查增强数据文件是否存在
        if os.path.exists(cache_file_path):
            try:
                # 读取增强数据
                ds = xr.open_dataset(cache_file_path)

                # 过滤站点ID
                available_stations = [
                    station
                    for station in gage_id_lst
                    if station in ds.coords.get("gage_id", [])
                ]
                if available_stations:
                    ds = ds.sel(gage_id=available_stations)

                # 应用时间范围过滤
                if t_range is not None and len(t_range) >= 2:
                    # 获取数据集的时间范围
                    ds_start_time = str(ds.time.values[0])
                    ds_end_time = str(ds.time.values[-1])

                    # 确定实际的切片范围
                    actual_start = max(t_range[0], ds_start_time)
                    actual_end = min(t_range[1], ds_end_time)

                    # 使用调整后的时间范围进行切片
                    ds = ds.sel(time=slice(actual_start, actual_end))

                    # 如果切片后的数据集为空，返回警告
                    if len(ds.time) == 0:
                        print(
                            f"Warning: The specified time range {t_range [0]} to {t_range [1]} does not overlap with the dataset time range {ds_start_time} to {ds_end_time}"
                        )

                # 应用变量过滤
                if var_lst is not None:
                    available_vars = [var for var in var_lst if var in ds.data_vars]
                    if available_vars:
                        ds = ds[available_vars]

                # 为变量添加必需的 units 属性
                for var_name in ds.data_vars:
                    if "units" not in ds[var_name].attrs:
                        # 根据变量类型添加合适的单位
                        if any(
                            keyword in var_name.lower()
                            for keyword in [
                                "flow",
                                "inflow",
                                "streamflow",
                                "gen_discharge",
                                "obs_discharge",
                            ]
                        ):  # 如果 var_name 包含任意一个关键词，执行这里的代码
                            ds[var_name].attrs["units"] = "m^3/s"  # 流量单位
                        elif (
                            "rain" in var_name.lower()
                            or "precipitation" in var_name.lower()
                        ):
                            ds[var_name].attrs["units"] = f"mm/{time_unit}"  # 降雨单位
                        elif "flood_event" in var_name.lower():
                            ds[var_name].attrs["units"] = "dimensionless"  # 无量纲
                        else:
                            ds[var_name].attrs["units"] = "unknown"  # 默认值

                print(f"Successfully read from augmented data cache: {cache_file_path}")
                return {time_unit: ds}

            except Exception as e:
                print(
                    f"Failed to read augmented data cache, fallback to original data: {e}"
                )

        # 如果增强数据不存在或读取失败，回退到原始数据
        result = self.read_ts_xrdataset(gage_id_lst, t_range, var_lst, **kwargs)

        # 为所有返回的数据集添加 units 属性
        for time_unit_key, ds in result.items():
            for var_name in ds.data_vars:
                if "units" not in ds[var_name].attrs:
                    # 根据变量类型添加合适的单位
                    if any(
                        keyword in var_name.lower()
                        for keyword in ["flow", "inflow", "streamflow"]
                    ):
                        ds[var_name].attrs["units"] = "m^3/s"  # 流量单位
                    elif (
                        "rain" in var_name.lower()
                        or "precipitation" in var_name.lower()
                    ):
                        ds[var_name].attrs["units"] = f"mm/{time_unit}"  # 降雨单位
                    elif "flood_event" in var_name.lower():
                        ds[var_name].attrs["units"] = "dimensionless"  # 无量纲
                    else:
                        ds[var_name].attrs["units"] = "unknown"  # 默认值

        return result

    def discover_augmented_files(
        self,
        augmented_files_dir: str,
        source_event: Optional[str] = None,
        modified_by: Optional[List[str]] = None,
        time_range: Optional[Tuple[str, str]] = None,
        latest_only: bool = False,
    ) -> List[Dict]:
        """
        发现增强数据文件的智能接口

        Args:
            augmented_files_dir: 增强数据文件目录
            source_event: 源事件名过滤 (如 "event_1994081520_1994081805")
            modified_by: 修改者列表过滤
            time_range: 修改时间范围过滤 ("2025-01-01", "2025-12-31")
            latest_only: 是否只返回每个源事件的最新版本

        Returns:
            List[Dict]: 文件信息列表，包含文件路径、元数据等
        """
        print(f"🔍 Search for augmented data files: {augmented_files_dir}")

        # 查找所有CSV文件
        csv_pattern = os.path.join(augmented_files_dir, "*.csv")
        all_files = glob.glob(csv_pattern)

        discovered_files = []

        for file_path in all_files:
            try:
                file_info = self._parse_augmented_file_info(file_path)

                # 应用过滤条件
                if source_event and file_info["source_event"] != source_event:
                    continue

                if modified_by and file_info["modified_by"] not in modified_by:
                    continue

                if time_range and file_info["modified_time"]:
                    file_time = datetime.strptime(
                        file_info["modified_time"], "%Y-%m-%d %H:%M:%S"
                    )
                    start_time = datetime.strptime(time_range[0], "%Y-%m-%d")
                    end_time = datetime.strptime(time_range[1], "%Y-%m-%d")
                    if not (start_time <= file_time <= end_time):
                        continue

                discovered_files.append(file_info)
            except Exception as e:
                print(f"   ⚠️ skip file {file_path}: {str(e)}")
                continue

        # 按源事件分组并选择最新版本
        if latest_only and discovered_files:
            latest_files = {}
            for file_info in discovered_files:
                source = file_info["source_event"]
                if source not in latest_files:
                    latest_files[source] = file_info
                else:
                    # 比较修改时间，选择最新的
                    current_time = file_info.get("modified_time") or ""
                    existing_time = latest_files[source].get("modified_time") or ""
                    if current_time > existing_time:
                        latest_files[source] = file_info

            discovered_files = list(latest_files.values())

        # 按修改时间排序，处理None值
        discovered_files.sort(key=lambda x: x.get("modified_time") or "", reverse=True)
        print(f"   ✅ Found {len (discovered_files)} files that meet the criteria")
        return discovered_files

    def _parse_augmented_file_info(self, file_path: str) -> Dict:
        """解析增强数据文件的信息"""
        filename = os.path.basename(file_path)

        # 解析文件名格式: xxx_modifiedbyXXX_atYYYYMMDDHHMMSS.csv
        name_pattern = r"(.+)_modifiedby(.+)_at(\d{14})\.csv"
        match = re.match(name_pattern, filename)
        file_info: Dict[str, Any] = {
            "file_path": file_path,
            "filename": filename,
            "source_event": None,
            "modified_by": None,
            "modified_time": None,
            "metadata": {},
        }

        if match:
            base_name = match.group(1)
            modifier = match.group(2)
            timestamp = match.group(3)

            file_info["source_event"] = base_name
            file_info["modified_by"] = modifier

            # 解析时间戳
            try:
                dt = datetime.strptime(timestamp, "%Y%m%d%H%M%S")
                file_info["modified_time"] = dt.strftime("%Y-%m-%d %H:%M:%S")
            except ValueError:
                pass
        # 读取文件头部的元数据
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            metadata = {}
            for line in lines:
                if line.startswith("#"):
                    line = line.strip("#").strip()
                    if ":" in line:
                        key, value = line.split(":", 1)
                        key_clean = key.strip().lower().replace(" ", "_")
                        metadata[key_clean] = value.strip()
                else:
                    break

            file_info["metadata"] = metadata

            # 从元数据中获取更多信息
            if "source" in metadata:
                file_info["source_event"] = metadata["source"].replace(".csv", "")

        except Exception as e:
            print(f"   ⚠️ Unable to read file header: {file_path}, {str(e)}")
        return file_info

    def process_augmented_files_by_discovery(
        self,
        station_ids: Union[str, List[str]],
        augmented_files_dir: str,
        source_event: Optional[str] = None,
        modified_by: Optional[List[str]] = None,
        time_range: Optional[Tuple[str, str]] = None,
        latest_only: bool = True,
        warmup_hours: int = 240,
        time_unit: str = "3h",
        score_threshold: int = 60,
        use_score_as_weight:bool = False,
    ) -> Optional[str]:
        """
        Process augmented data files based on file discovery.

        Parameters
        ----------
        station_ids : Union[str, List[str]]
            Station ID or list of station IDs.
        augmented_files_dir : str
            Directory containing augmented data files.
        source_event : Optional[str], optional
            Filter by source event name.
        modified_by : Optional[List[str]], optional
            Filter by list of modifiers.
        time_range : Optional[Tuple[str, str]], optional
            Filter by modification time range.
        latest_only : bool, optional
            Whether to process only the latest version for each source event.
        warmup_hours : int, optional
            Number of warmup hours.
        time_unit : str, optional
            Time unit.
        score_threshold : int, optional
            Score threshold (default: 60). Behavior depends on use_score_as_weight:
            - If use_score_as_weight=False: Only events with Score >= score_threshold are included (filtering mode)
            - If use_score_as_weight=True: Events with Score < score_threshold get flood_event=1,
              events with Score >= score_threshold get flood_event=score (weighting mode)
        use_score_as_weight : bool, optional
            If True, use score-based weighting mode where flood_event stores actual scores (1-100)
            instead of binary values (0/1). Default is False (threshold filtering mode).
            - False (filtering): flood_event=0/1, only score>=threshold included
            - True (weighting): flood_event=-1/0 (non-flood), 1 (low-quality), or score (high-quality)

        Returns
        -------
        Optional[str]
            Path to the cache file, or None if processing fails.
        """
        # 兼容单个站点的情况
        if isinstance(station_ids, str):
            station_ids = [station_ids]

        # 发现可用文件
        discovered_files = self.discover_augmented_files(
            augmented_files_dir=augmented_files_dir,
            source_event=source_event,
            modified_by=modified_by,
            time_range=time_range,
            latest_only=latest_only,
        )

        if not discovered_files:
            print("❌ No augmented data files were found")
            return None
        print(f"🔄 Prepare to process {len (discovered_files)} augmented data files:")

        # 创建文件路径列表，用于现有的处理方法
        file_paths = [file_info["file_path"] for file_info in discovered_files]
        # 调用现有的处理方法，传递 use_score_as_weight 参数
        return self._process_augmented_files_by_paths(
            station_ids=station_ids,
            file_paths=file_paths,
            warmup_hours=warmup_hours,
            time_unit=time_unit,
            score_threshold=score_threshold,
            use_score_as_weight=use_score_as_weight,  # 新增
        )

    def _process_augmented_files_by_paths(
        self,
        station_ids: Union[str, List[str]],
        file_paths: List[str],
        warmup_hours: int = 240,
        time_unit: str = "3h",
        score_threshold: int = 60,
        use_score_as_weight: bool = False,  # 新增参数
    ) -> Optional[str]:
        """
        处理增强数据文件基于文件路径（结构化数组版）- 解决字典保存问题

        Parameters
        ----------
        station_ids : Union[str, List[str]]
            站点ID或站点ID列表
        file_paths : List[str]
            要处理的文件路径列表
        warmup_hours : int, optional
            预热小时数，默认240
        time_unit : str, optional
            时间单位，默认"3h"
        score_threshold : int, optional
            分数阈值，默认60。根据 use_score_as_weight 有不同行为
        use_score_as_weight : bool, optional
            是否使用评分加权模式，默认False（阈值过滤模式）

        Returns
        -------
        Optional[str]
            缓存文件路径，处理失败返回None
        """
        # 兼容单个站点的情况
        if isinstance(station_ids, str):
            station_ids = [station_ids]

        all_datasets = []
        processed_count = 0

        # 对每个站点处理增强数据
        for station_id in station_ids:
            print(f"🔄🔄 Processing station: {station_id}")
            station_timeseries = []
            total_files = len(file_paths)  # 总文件数
            processed_files = 0  # 成功处理的文件数

            for file_path in file_paths:
                try:
                    print(f"   🔄🔄 Processing file: {os.path.basename(file_path)}")

                    # 解析增强文件的元信息
                    metadata = self.parse_augmented_file_metadata(file_path)
                    if not metadata:
                        print(f"      ⚠⚠⚠️ Skip file {file_path}: unable to parse metadata")
                        continue

                    # 跳过2026+无评分的文件
                    if "Score" in metadata and metadata["Score"] is None:
                        print(f"      ⚠⚠⚠️ Skip unscored augmented data file: {os.path.basename(file_path)}")
                        continue

                    # 根据模式处理评分
                    if "Score" in metadata:
                        score = int(metadata["Score"])
                        if use_score_as_weight:
                            # 评分加权模式：所有数据都接受，稍后在 flood_event 中存储评分
                            print(f"      ✅ Accept file {os.path.basename(file_path)}: Score {score} (weighting mode)")
                        else:
                            # 阈值过滤模式：只接受 >= threshold 的数据
                            if score < score_threshold:
                                print(f"      ⚠⚠⚠️ Skip file {os.path.basename(file_path)}: Score {score} < threshold {score_threshold}")
                                continue
                            else:
                                print(f"      ✅ Accept file {os.path.basename(file_path)}: Score {score} >= threshold {score_threshold}")
                    else:
                        # 真实数据（无Score字段）默认通过
                        print(f"      ✅ Accept file {os.path.basename(file_path)}: Real data (no score)")

                    # 检查CSV文件是否已包含flood_event列
                    try:
                        sample_df = pd.read_csv(file_path, comment="#", nrows=1)
                        has_flood_event = "flood_event" in sample_df.columns
                    except Exception as e:
                        print(f"      ⚠⚠⚠️ Skip file {file_path}: unable to read CSV header - {str(e)}")
                        continue

                    # 仅当CSV不包含flood_event列时，才获取外部预热期数据
                    if has_flood_event:
                        warmup_df = pd.DataFrame()
                    else:
                        # 获取预热期数据
                        warmup_df = self.get_warmup_period_data(
                            original_start_time=metadata.get("original_start_time", ""),
                            original_end_time=metadata.get("original_end_time", ""),
                            station_id=station_id,
                            warmup_hours=warmup_hours,
                        )

                        if warmup_df is None:
                            print(f"      ⚠⚠⚠️ Skip file {file_path}: unable to get warmup data")
                            continue

                    # 调整预热期时间并拼接增强数据
                    timeseries_df = self.concatenate_warmup_and_augmented_data(
                        warmup_df,
                        file_path,
                        use_score_as_weight=use_score_as_weight,  # 新增
                        score_threshold=score_threshold,  # 新增
                    )

                    if timeseries_df is not None and len(timeseries_df) > 0:
                        station_timeseries.append(timeseries_df)
                        processed_files += 1  # 成功处理文件计数
                    else:
                        print(f"      ⚠⚠⚠️ Skip file {file_path}: The processed data is empty")

                except Exception as e:
                    print(f"      ❌❌ Processing file failed {file_path}: {str(e)}")
                    continue

            # 如果该站点有数据，则合并并转换为xarray Dataset
            if station_timeseries:
                print(f"\n   🔄🔄 Merge time series data of {station_id}...")
                print(f"   📊 Successfully merged {processed_files}/{total_files} augmented files")
                combined_df = pd.concat(station_timeseries, ignore_index=True)
                combined_df = combined_df.sort_values("time").reset_index(drop=True)

                # 重命名列 gen_discharge -> inflow
                combined_df = self.rename_dataframe_columns(combined_df)

                # 转换为xarray Dataset
                station_ds = self.create_xarray_dataset_from_augdf(combined_df, station_id, time_unit)
                all_datasets.append(station_ds)
                processed_count += 1
                print(f"   ✅ {station_id} processing completed: {len(combined_df)} records\n")
            else:
                print(f"   ⚠⚠⚠️  {station_id} No data available")

        if not all_datasets:
            print("❌❌ Failed to process data from any basin successfully")
            return None

        print(f"✅ Successfully processed {processed_count} basins")

        # 合并所有站点的数据集
        if len(all_datasets) == 1:
            final_ds = all_datasets[0]
        else:
            print("🔄🔄 Merge datasets from multiple basins...")
            final_ds = xr.concat(all_datasets, dim="basin")  # 修正：使用basin而非gage_id

        # 添加数据集属性
        final_ds.attrs.update({
            'processing_timestamp': datetime.now().isoformat(),
            'data_source': 'augmented_flood_events',
            'use_score_as_weight': int(use_score_as_weight),  # 转换 boolean 为 int (NetCDF 不支持 boolean 类型)
        })

        # 保存到缓存
        print("🔄🔄 Save augmented data to cache...")
        cache_file_path = self.save_augmented_timeseries_to_cache(final_ds, station_ids, time_unit)
        
        if cache_file_path:
            print(f"✅ Successfully saved to: {cache_file_path}")
        else:
            print("❌ Failed to save cache file")
        
        return cache_file_path

    def get_user_contributions_summary(self, augmented_files_dir: str) -> pd.DataFrame:
        """获取用户贡献统计"""
        discovered_files = self.discover_augmented_files(augmented_files_dir)
        if not discovered_files:
            return pd.DataFrame()
        summary_data = []
        user_stats = {}
        for file_info in discovered_files:
            user = file_info.get("modified_by", "unknown")
            source = file_info.get("source_event", "unknown")

            if user not in user_stats:
                user_stats[user] = {
                    "user": user,
                    "total_files": 0,
                    "unique_events": set(),
                    "latest_modification": None,
                }

            user_stats[user]["total_files"] += 1
            user_stats[user]["unique_events"].add(source)

            mod_time = file_info.get("modified_time")
            if mod_time:
                latest = user_stats[user]["latest_modification"]
                if latest is None or mod_time > latest:
                    user_stats[user]["latest_modification"] = mod_time

        for user, stats in user_stats.items():
            summary_data.append(
                {
                    "user": user,
                    "total_files": stats["total_files"],
                    "unique_events": len(stats["unique_events"]),
                    "latest_modification": stats["latest_modification"],
                }
            )
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values("total_files", ascending=False)
        return summary_df

    def check_event_data_nan(
        self,
        all_event_data: List[Dict],
        exclude_warmup: bool = True,
    ):
        """
        Check for NaN values in rainfall and runoff data for all flood events.

        This method leverages the class's attributes (net_rain_key, obs_flow_key,
        warmup_length) and can optionally exclude warmup period from NaN checking.

        Parameters
        ----------
        all_event_data : list of dict
            List of event dictionaries, each containing net rainfall, runoff, filepath, etc.
        exclude_warmup : bool, optional
            Whether to exclude warmup period data from NaN checking, by default True.
            When True, only checks data points where flood_event_markers > 0 or
            excludes the first warmup_length data points if markers are not available.

        Raises
        ------
        ValueError
            If any NaN values are found in the non-warmup data, raises an exception
            and prints detailed information.

        Notes
        -----
        This method uses the class attributes:
        - self.net_rain_key: Key name for net rainfall data
        - self.obs_flow_key: Key name for observed flow data
        - self.warmup_length: Number of warmup time steps

        When exclude_warmup=True:
        1. If 'flood_event_markers' exists in event data, only checks where markers > 0
        2. Otherwise, skips the first self.warmup_length data points
        """
        for event in all_event_data:
            event_name = event.get("filepath", "unknown")
            p_eff = event.get(self.net_rain_key)
            q_obs = event.get(self.obs_flow_key)

            if p_eff is None or q_obs is None:
                continue

            # Convert to numpy arrays if needed
            p_eff = np.array(p_eff)
            q_obs = np.array(q_obs)

            # Determine which indices to check based on exclude_warmup setting
            if exclude_warmup:
                flood_event_markers = event.get("flood_event_markers")

                if flood_event_markers is not None:
                    # Use flood_event_markers to identify non-warmup data
                    flood_event_markers = np.array(flood_event_markers)
                    check_indices = flood_event_markers > 0

                    # Apply mask to data arrays
                    p_eff_to_check = p_eff[check_indices]
                    q_obs_to_check = q_obs[check_indices]

                    # Get the actual indices for error reporting
                    original_indices = np.arange(len(p_eff))[check_indices]
                else:
                    # Fallback: exclude first warmup_length points
                    start_idx = self.warmup_length
                    p_eff_to_check = p_eff[start_idx:]
                    q_obs_to_check = q_obs[start_idx:]

                    # Get the actual indices for error reporting
                    original_indices = np.arange(start_idx, len(p_eff))
            else:
                # Check all data points
                p_eff_to_check = p_eff
                q_obs_to_check = q_obs
                original_indices = np.arange(len(p_eff))

            # Check for NaN in net rain
            if np.any(np.isnan(p_eff_to_check)):
                nan_mask = np.isnan(p_eff_to_check)
                nan_indices = original_indices[nan_mask]
                print(
                    f"❌ Event {event_name} has NaN in {self.net_rain_key} at index: {nan_indices}"
                )
                raise ValueError(
                    f"Event {event_name} has NaN in {self.net_rain_key} at index {nan_indices}"
                )

            # Check for NaN in observed flow
            if np.any(np.isnan(q_obs_to_check)):
                nan_mask = np.isnan(q_obs_to_check)
                nan_indices = original_indices[nan_mask]
                print(
                    f"❌ Event {event_name} has NaN in {self.obs_flow_key} at index: {nan_indices}"
                )
                raise ValueError(
                    f"Event {event_name} has NaN in {self.obs_flow_key} at index {nan_indices}"
                )


def _calculate_event_characteristics(
    event: Dict,
    delta_t_hours: float = 3.0,
    net_rain_key: str = "P_eff",
    obs_flow_key: str = "Q_obs_eff",
) -> Dict:
    """
    计算洪水事件的详细特征指标，用于画图和分析

    Parameters
    ----------
        event: dict
            事件字典，包含净雨和径流数组
        delta_t_hours: float
            时段长度（小时），默认3小时
        net_rain_key: str
            净雨数据的键名，默认"P_eff"
        obs_flow_key: str
            观测流量数据的键名，默认"Q_obs_eff"

    Returns
    -------
        Dict: 包含计算出的水文特征指标

    Calculated metrics:
        - peak_obs: 洪峰流量 (m³/s)
        - runoff_volume_m3: 洪量 (m³)
        - runoff_duration_hours: 洪水历时 (小时)
        - total_net_rain: 总净雨量 (mm)
        - lag_time_hours: 洪峰雨峰延迟 (小时)
    """
    try:
        # 提取数据
        net_rain = event.get(net_rain_key, [])
        direct_runoff = event.get(obs_flow_key, [])

        net_rain = np.array(net_rain)
        direct_runoff = np.array(direct_runoff)

        # 转换为秒
        delta_t_seconds = delta_t_hours * 3600.0

        # 1. 计算洪峰流量
        peak_obs = np.max(direct_runoff)
        if peak_obs < 1e-6:
            return None

        # 2. 计算洪量 (m³)
        runoff_volume_m3 = np.sum(direct_runoff) * delta_t_seconds

        # 3. 计算洪水历时 (小时)
        runoff_indices = np.where(direct_runoff > 1e-6)[0]
        if len(runoff_indices) < 2:
            return None
        runoff_duration_hours = (
            runoff_indices[-1] - runoff_indices[0] + 1
        ) * delta_t_hours

        # 4. 计算总净雨量 (mm)
        total_net_rain = np.sum(net_rain)

        # 5. 计算洪峰雨峰延迟 (小时)
        t_peak_flow_idx = np.argmax(direct_runoff)
        t_peak_rain_idx = np.argmax(net_rain)
        lag_time_hours = (t_peak_flow_idx - t_peak_rain_idx) * delta_t_hours

        # 6. 计算有效降雨时段数
        m_eff = len(net_rain)

        # 7. 计算径流时段数
        n_obs = len(direct_runoff)

        # 8. 计算单位线长度
        n_specific = n_obs - m_eff + 1

        # 返回计算结果
        characteristics = {
            "peak_obs": peak_obs,  # 洪峰流量 (m³/s)
            "runoff_volume_m3": runoff_volume_m3,  # 洪量 (m³)
            "runoff_duration_hours": runoff_duration_hours,  # 洪水历时 (小时)
            "total_net_rain": total_net_rain,  # 总净雨量 (mm)
            "lag_time_hours": lag_time_hours,  # 洪峰雨峰延迟 (小时)
            "m_eff": m_eff,  # 有效降雨时段数
            "n_obs": n_obs,  # 径流时段数
            "n_specific": n_specific,  # 单位线长度
            "delta_t_hours": delta_t_hours,  # 时段长度
        }

        return characteristics

    except Exception as e:
        print(f"Error calculating event features: {e}")
        return None


def calculate_events_characteristics(
    events: List[Dict],
    delta_t_hours: float = 3.0,
    net_rain_key: str = "P_eff",
    obs_flow_key: str = "Q_obs_eff",
) -> List[Dict]:
    """
    批量计算多个洪水事件的特征指标

    Args:
        events: 事件列表，每个事件包含净雨和径流数组
        delta_t_hours: 时段长度（小时），默认3小时
        net_rain_key: 净雨数据的键名，默认"P_eff"
        obs_flow_key: 观测流量数据的键名，默认"Q_obs_eff"

    Returns:
        List[Dict]: 包含计算出的水文特征指标的事件列表
    """
    enhanced_events = []

    for i, event in enumerate(events):
        # 计算特征指标
        characteristics = _calculate_event_characteristics(
            event, delta_t_hours, net_rain_key, obs_flow_key
        )

        if characteristics:
            # 将特征指标添加到原事件字典中
            enhanced_event = event.copy()
            enhanced_event.update(characteristics)
            enhanced_events.append(enhanced_event)
        else:
            print(f"⚠️ 事件 {i+1} 特征计算失败，跳过")

    return enhanced_events
