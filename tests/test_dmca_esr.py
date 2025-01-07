"""
Author: liutiaxqabs 1498093445@qq.com
Date: 2024-05-15 10:26:29
LastEditors: Wenyu Ouyang
LastEditTime: 2025-01-07 21:17:44
FilePath: \hydrodatasource\tests\test_dmca_esr.py
Description: TODO: This test file need to be refactored
"""

import os
import pandas as pd
import numpy as np
from pint import UnitRegistry

from hydrodataset import Camels

from hydrodatasource.configs.config import SETTING
from hydrodatasource.processor.dmca_esr import *
from hydrodatasource.utils.utils import streamflow_unit_conv


def test_rainfall_runoff_event_identify():
    camels = Camels(
        os.path.join(
            SETTING["local_data_path"]["datasets-origin"], "camels", "camels_us"
        )
    )
    gage_ids = camels.read_object_ids()
    ureg = UnitRegistry()

    rain = camels.read_ts_xrdataset(
        gage_ids[:1], ["1980-01-01", "2015-01-01"], var_lst=["prcp"]
    )
    flow = camels.read_ts_xrdataset(
        gage_ids[:1], ["1980-01-01", "2015-01-01"], var_lst=["streamflow"]
    )
    # trans unit to mm/day
    basin_area = camels.read_area(gage_ids[:1])
    r_mmd = streamflow_unit_conv(flow, basin_area)
    flow_threshold = streamflow_unit_conv(
        np.array([100]) * ureg.m**3 / ureg.s,
        basin_area.isel(basin=0).to_array().to_numpy() * ureg.km**2,
        target_unit="mm/h",
    )
    flood_events = rainfall_runoff_event_identify(
        rain["prcp"].isel(basin=0).to_series(),
        r_mmd["streamflow"].isel(basin=0).to_series(),
        flow_threshold=flow_threshold[0],
    )
    assert flood_events["BEGINNING_RAIN"].shape[0] == flood_events["END_RAIN"].shape[0]


def test_biliu_rain_flow_division():
    # rain和flow之间的索引要尽量“对齐”
    # 2014.1.1 00:00:00-2022.9.1 00:00:00
    # 读取并处理 rain.csv 文件
    filtered_rain_aver_df = pd.read_csv(
        os.path.join("/home/liutianxv1/0515场次划分/2010-2023流域平均数据.csv"),
        engine="c",
    ).set_index("TM")

    # 将时间列转换为 datetime 对象，并去重
    filtered_rain_aver_df.index = pd.to_datetime(
        filtered_rain_aver_df.index, errors="coerce"
    )
    filtered_rain_aver_df = filtered_rain_aver_df[
        ~filtered_rain_aver_df.index.duplicated(keep="first")
    ]

    filtered_rain_aver_array = filtered_rain_aver_df["rain"].to_numpy()

    # 读取并处理 inq.csv 文件
    flow_mm_h_df = pd.read_csv(
        os.path.join("/home/liutianxv1/0515场次划分/zq_CHN_songliao_21401550.csv"),
        engine="c",
    ).set_index("TM")

    # 将时间列转换为 datetime 对象，并去重
    flow_mm_h_df.index = pd.to_datetime(flow_mm_h_df.index, errors="coerce")
    flow_mm_h_df = flow_mm_h_df[~flow_mm_h_df.index.duplicated(keep="first")]

    flow = flow_mm_h_df["INQ"]
    biliu_area = 2097
    flow_mm_h = flow.apply(lambda x: x * 3.6 / biliu_area)

    # 确保两个 DataFrame 的索引对齐
    common_index = filtered_rain_aver_df.index.intersection(flow_mm_h_df.index)
    filtered_rain_aver_df = filtered_rain_aver_df.loc[common_index]
    flow_mm_h_df = flow_mm_h_df.loc[common_index]

    time = filtered_rain_aver_df.index

    # 如果需要将时间格式重新转换为字符串，可以使用strftime
    time = pd.to_datetime(time).strftime("%Y-%m-%d %H:%M:%S")

    rain_min = 0.02
    max_window = 100
    Tr, fluct_rain_Tr, fluct_flow_Tr, fluct_bivariate_Tr = (
        step1_step2_tr_and_fluctuations_timeseries(
            filtered_rain_aver_array, flow_mm_h, rain_min, max_window
        )
    )

    beginning_core, end_core = step3_core_identification(fluct_bivariate_Tr)
    end_rain = step4_end_rain_events(
        beginning_core, end_core, filtered_rain_aver_array, fluct_rain_Tr, rain_min
    )
    beginning_rain = step5_beginning_rain_events(
        beginning_core, end_rain, filtered_rain_aver_array, fluct_rain_Tr, rain_min
    )
    beginning_rain_checked, end_rain_checked, beginning_core, end_core = (
        step6_checks_on_rain_events(
            beginning_rain,
            end_rain,
            filtered_rain_aver_array,
            rain_min,
            beginning_core,
            end_core,
        )
    )
    end_flow = step7_end_flow_events(
        end_rain_checked,
        beginning_core,
        end_core,
        filtered_rain_aver_array,
        fluct_rain_Tr,
        fluct_flow_Tr,
        Tr,
    )
    beginning_flow = step8_beginning_flow_events(
        beginning_rain_checked,
        end_rain_checked,
        filtered_rain_aver_array,
        beginning_core,
        fluct_rain_Tr,
        fluct_flow_Tr,
    )
    (
        beginning_rain_ungrouped,
        end_rain_ungrouped,
        beginning_flow_ungrouped,
        end_flow_ungrouped,
    ) = step9_checks_on_flow_events(
        beginning_rain_checked,
        end_rain_checked,
        beginning_flow,
        end_flow,
        fluct_flow_Tr,
    )
    BEGINNING_RAIN, END_RAIN, BEGINNING_FLOW, END_FLOW = (
        step10_checks_on_overlapping_events(
            beginning_rain_ungrouped,
            end_rain_ungrouped,
            beginning_flow_ungrouped,
            end_flow_ungrouped,
            time,
        )
    )
    print(len(BEGINNING_RAIN), len(END_RAIN), len(BEGINNING_FLOW), len(END_FLOW))
    # print('_________________________')
    # print('_________________________')
    # print(BEGINNING_FLOW, END_FLOW)
    multiple = 1
    flag = 0
    flow_max = 100
    biliu_area = 2097
    flow_max_mm_h = flow_max * 3.6 / biliu_area
    duration_max = 2400
    (
        DURATION_RAIN,
        VOLUME_RAIN,
        DURATION_RUNOFF,
        VULUME_RUNOFF,
        RUNOFF_RATIO,
        BEGINNING_RAIN,
        END_RAIN,
        BEGINNING_FLOW,
        END_FLOW,
    ) = event_analysis(
        BEGINNING_RAIN,
        END_RAIN,
        BEGINNING_FLOW,
        END_FLOW,
        filtered_rain_aver_array,
        flow_mm_h,
        time,
        flag,
        multiple,
        flow_max_mm_h,
        duration_max,
    )
    biliu_division = pd.DataFrame(
        {
            "BEGINNING_RAIN": BEGINNING_RAIN,
            "END_RAIN": END_RAIN,
            "DURATION_RAIN": DURATION_RAIN,
            "BEGINNING_FLOW": BEGINNING_FLOW,
            "END_FLOW": END_FLOW,
            "DURATION_RUNOFF": DURATION_RUNOFF,
            "VOLUME_RAIN": VOLUME_RAIN,
            "VULUME_RUNOFF": VULUME_RUNOFF,
            "RUNOFF_RATIO": RUNOFF_RATIO,
        }
    )

    biliu_division.to_csv(
        os.path.join(
            "/home/liutianxv1/0515场次划分/biliu_flow_division.csv",
        ),
        index=False,
    )
