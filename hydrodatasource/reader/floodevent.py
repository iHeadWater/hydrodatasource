"""
Author: Wenyu Ouyang
Date: 2025-01-19 18:05:00
LastEditTime: 2025-07-17 16:22:06
LastEditors: Wenyu Ouyang
Description: 流域场次数据处理类 - 继承自SelfMadeHydroDataset
FilePath: /hydrodatasource/hydrodatasource/reader/floodevent.py
Copyright (c) 2023-2026 Wenyu Ouyang. All rights reserved.
"""

import pandas as pd
import numpy as np
import os
from typing import List, Dict, Optional, Tuple
from hydrodatasource.configs.data_consts import FLOOD_EVENT_VARS
from hydrodatasource.utils.utils import streamflow_unit_conv
from hydrodatasource.reader.data_source import SelfMadeHydroDataset


class FloodEventDatasource(SelfMadeHydroDataset):
    """
    流域场次数据处理类

    继承自SelfMadeHydroDataset，专门用于处理到逐个洪水场次数据，
    包括读取流域面积、单位转换、场次提取等功能。
    """

    def __init__(
        self,
        data_path: str,
        dataset_name: str = "songliaorrevents",
        time_unit: Optional[List[str]] = None,
        flow_unit: str = "mm/3h",
        **kwargs,
    ):
        """
        初始化流域场次数据集

        Args:
            data_path: 数据路径
            dataset_name: 数据集名称
            time_unit: 时间单位列表，默认为["3h"]
            flow_unit: 径流单位，默认为"mm/3h"
            **kwargs: 其他参数传递给父类
        """
        if time_unit is None:
            time_unit = ["3h"]
        # sometimes we load the data with different flow unit
        # so we need to store the flow unit
        self.flow_unit = flow_unit
        super().__init__(
            data_path=data_path,
            download=False,
            time_unit=time_unit,
            dataset_name=dataset_name,
            **kwargs,
        )

    def extract_flood_events(
        self, df: pd.DataFrame
    ) -> List[Tuple[np.ndarray, np.ndarray, str]]:
        """
        从数据框中提取洪水事件，返回净雨、径流数组和洪峰日期

        Args:
            df: 站点数据框
            station_id: 站点ID（用于打印信息）

        Returns:
            List[Tuple[np.ndarray, np.ndarray, str]]: (净雨数组, 径流数组, 洪峰日期) 列表
        """
        events = []
        # 找到连续的flood_event > 0区间
        flood_mask = df["flood_event"] > 0

        if not flood_mask.any():
            return events

        # 找连续区间
        in_event = False
        start_idx = None

        for idx, is_flood in enumerate(flood_mask):
            if is_flood and not in_event:
                start_idx = idx
                in_event = True
            elif not is_flood and in_event:
                # 事件结束，提取数据
                event_data = df.iloc[start_idx:idx]
                net_rain = event_data["net_rain"].values
                inflow = event_data["inflow"].values
                event_times = event_data["time"].values

                # 基本验证
                if len(net_rain) > 0 and len(inflow) > 0 and np.nansum(inflow) > 1e-6:
                    # 获取场次开始和结束时间
                    start_time = event_times[0]
                    end_time = event_times[-1]

                    # 转换为十位数字格式 (YYYYMMDDHH)
                    def time_to_ten_digits(time_obj):
                        """将时间对象转换为十位数字格式 YYYYMMDDHH"""
                        if isinstance(time_obj, np.datetime64):
                            # 如果是numpy datetime64对象
                            return (
                                time_obj.astype("datetime64[h]")
                                .astype(str)
                                .replace("-", "")
                                .replace("T", "")
                                .replace(":", "")
                            )
                        elif hasattr(time_obj, "strftime"):
                            # 如果是datetime对象
                            return time_obj.strftime("%Y%m%d%H")
                        else:
                            # 如果是字符串，尝试解析
                            try:
                                from datetime import datetime

                                if isinstance(time_obj, str):
                                    dt = datetime.fromisoformat(
                                        time_obj.replace("Z", "+00:00")
                                    )
                                    return dt.strftime("%Y%m%d%H")
                                else:
                                    return "0000000000"  # 默认值
                            except:
                                return "0000000000"  # 默认值

                    start_digits = time_to_ten_digits(start_time)
                    end_digits = time_to_ten_digits(end_time)

                    # 组合成场次名称：起始时间_结束时间
                    event_name = f"{start_digits}_{end_digits}"

                    events.append((net_rain, inflow, event_name))

                in_event = False
        return events

    def create_event_dict(
        self,
        net_rain: np.ndarray,
        inflow: np.ndarray,
        event_name: str,
        include_peak_obs: bool = True,
    ) -> Optional[Dict]:
        """
        将净雨和径流数组转换为标准事件字典格式

        Parameters
        ----------
        net_rain: np.ndarray
            净雨数组
        inflow: np.ndarray
            径流数组
        event_name: str
            洪峰日期（8位数字格式）
        include_peak_obs: bool
            是否包含洪峰观测值

        Returns
        -------
            Dict: 标准格式的事件字典，与uh_utils.py完全兼容
        """
        try:
            # 计算有效降雨时段数
            valid_rain_mask = ~np.isnan(net_rain) & (net_rain > 0)
            m_eff = np.sum(valid_rain_mask)

            if m_eff == 0:
                return None

            # 验证径流数据
            if np.nansum(inflow) < 1e-6:
                return None

            # 创建标准格式字典（与uh_utils.py期望的key完全一致）
            event_dict = {
                FLOOD_EVENT_VARS["NET_RAIN"]: net_rain,  # 有效降雨（净雨）
                FLOOD_EVENT_VARS["OBS_FLOW"]: inflow,  # 观测径流
                "m_eff": m_eff,  # 有效降雨时段数
                "n_specific": len(net_rain),  # 单位线长度
                "filepath": f"event_{event_name}.csv",  # 添加filepath字段避免KeyError
            }

            # 添加洪峰观测值
            if include_peak_obs:
                peak_flow = np.nanmax(inflow)
                if peak_flow < 1e-6:
                    return None
                event_dict["peak_obs"] = peak_flow

            return event_dict

        except Exception:
            return None

    def _load_1basin_flood_events(
        self,
        station_id: Optional[str] = None,
        include_peak_obs: bool = True,
        verbose: bool = True,
    ) -> Optional[List[Dict]]:
        """
        加载洪水事件数据

        Parameters
        ----------
        station_id:
            指定站点ID，如果为None则处理所有站点
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
            try:
                basin_area_km2 = self.read_area([station_id])
                if verbose:
                    print(f"📊 读取到流域面积: {basin_area_km2} km²")
            except Exception as e:
                if verbose:
                    print(f"⚠️ 无法读取流域面积: {str(e)}")

        try:
            if verbose:
                print("🔄 正在加载洪水事件数据...")
                if station_id:
                    print(f"   指定站点: {station_id}")

            all_events = []
            total_events = 0

            xr_ds = self.read_ts_xrdataset(
                gage_id_lst=[station_id],
                t_range=["1960-01-01", "2024-12-31"],
                var_lst=["inflow", "net_rain", "flood_event"],
                # recache=True,
            )["3h"]
            if self.flow_unit == "mm/3h":
                xr_ds["inflow"] = streamflow_unit_conv(
                    xr_ds[["inflow"]], basin_area_km2, target_unit="mm/3h"
                )["inflow"]
            elif self.flow_unit == "m^3/s":
                pass
            else:
                raise ValueError(f"Unsupported flow unit: {self.flow_unit}")
            df = xr_ds.to_dataframe()
            if df is None:
                return None

            # 提取洪水事件
            flood_events = self.extract_flood_events(df.loc[station_id].reset_index())

            if not flood_events:
                if verbose:
                    print(f"  ⚠️  {station_id}: 没有找到有效洪水事件")
                return None

            # 转换为标准格式
            station_event_count = 0
            for net_rain, inflow, event_name in flood_events:
                event_dict = self.create_event_dict(
                    net_rain, inflow, event_name, include_peak_obs
                )
                if event_dict is not None:
                    all_events.append(event_dict)
                    station_event_count += 1

            if verbose and station_event_count > 0:
                print(f"  ✅ {station_id}: 成功处理 {station_event_count} 个洪水事件")
                total_events += station_event_count

            if not all_events:
                if verbose:
                    print("❌ 没有成功处理的洪水事件数据")
                return None

            if verbose:
                print(f"✅ 总共成功加载 {len(all_events)} 个洪水事件")

            return all_events

        except Exception as e:
            if verbose:
                print(f"❌ 加载洪水事件数据时发生错误: {str(e)}")
            return None


def _calculate_event_characteristics(event: Dict, delta_t_hours: float = 3.0) -> Dict:
    """
    计算洪水事件的详细特征指标，用于画图和分析

    Parameters
    ----------
        event: dict
            事件字典，包含 'P_eff' (净雨) 和 'Q_obs_eff' (径流) 数组
        delta_t_hours: float
            时段长度（小时），默认3小时

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
        net_rain = event.get(FLOOD_EVENT_VARS["NET_RAIN"], [])
        direct_runoff = event.get(FLOOD_EVENT_VARS["OBS_FLOW"], [])

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
        print(f"计算事件特征时出错: {e}")
        return None


def calculate_events_characteristics(
    events: List[Dict], delta_t_hours: float = 3.0
) -> List[Dict]:
    """
    批量计算多个洪水事件的特征指标

    Args:
        events: 事件列表，每个事件包含 'P_eff' 和 'Q_obs_eff' 数组
        delta_t_hours: 时段长度（小时），默认3小时

    Returns:
        List[Dict]: 包含计算出的水文特征指标的事件列表
    """
    enhanced_events = []

    for i, event in enumerate(events):
        # 计算特征指标
        characteristics = _calculate_event_characteristics(event, delta_t_hours)

        if characteristics is not None:
            # 将特征指标添加到原事件字典中
            enhanced_event = event.copy()
            enhanced_event.update(characteristics)
            enhanced_events.append(enhanced_event)
        else:
            print(f"⚠️ 事件 {i+1} 特征计算失败，跳过")

    return enhanced_events


def load_and_preprocess_events_unified(
    data_dir: str,
    station_id: Optional[str] = None,
    include_peak_obs: bool = True,
    verbose: bool = True,
    flow_unit: str = "mm/3h",
) -> Optional[List[Dict]]:
    """
    向后兼容的统一接口函数

    Args:
        data_source: 数据文件夹路径
        station_id: 流域站点ID（可选）
        include_peak_obs: 是否包含洪峰观测值
        verbose: 是否打印详细信息
        recache: 是否重新缓存数据，默认为False

    Returns:
        List[Dict]: 标准格式的事件字典列表，与现有单位线算法完全兼容
    """
    # 创建数据集实例
    dataset = FloodEventDatasource(
        data_dir,
        flow_unit=flow_unit,
        trange4cache=["1960-01-01 02", "2024-12-31 23"],
    )
    return dataset._load_1basin_flood_events(station_id, include_peak_obs, verbose)


def check_event_data_nan(all_event_data: List[Dict]):
    """
    检查所有洪水事件数据中的降雨和径流是否有空值，若有则报错并打印详细信息。
    Args:
        all_event_data: 事件字典列表（每个字典包含P_eff、Q_obs_eff、filepath等）
    Raises:
        ValueError: 如果发现空值，抛出异常并打印详细信息
    """
    for event in all_event_data:
        event_name = event.get("filepath", "unknown")
        p_eff = event.get(FLOOD_EVENT_VARS["NET_RAIN"])
        q_obs = event.get(FLOOD_EVENT_VARS["OBS_FLOW"])
        # 检查降雨
        if p_eff is not None and np.any(np.isnan(p_eff)):
            nan_idx = np.where(np.isnan(p_eff))[0]
            print(f"❌ 场次 {event_name} 的 P_eff 存在空值，索引: {nan_idx}")
            raise ValueError(f"Event {event_name} has NaN in P_eff at index {nan_idx}")
        # 检查径流
        if q_obs is not None and np.any(np.isnan(q_obs)):
            nan_idx = np.where(np.isnan(q_obs))[0]
            print(
                f"❌ 场次 {event_name} 的 {FLOOD_EVENT_VARS['OBS_FLOW']} 存在空值，索引: {nan_idx}"
            )
            raise ValueError(
                f"Event {event_name} has NaN in {FLOOD_EVENT_VARS['OBS_FLOW']} at index {nan_idx}"
            )
