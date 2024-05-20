"""
Author: Yang Wang, Jingyi Wang, and Wenyu Ouyang
Date: 2023-10-28 09:23:22
LastEditTime: 2024-03-28 08:40:17
LastEditors: Wenyu Ouyang
Description: Test for rainfall-runoff event identification
FilePath: \hydrodatasource\tests\test_rr_event_iden.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import os
import numpy as np
from pint import UnitRegistry

from hydrodataset import Camels

from hydrodatasource.configs.config import SETTING
from hydrodatasource.cleaner.dmca_esr import rainfall_runoff_event_identify
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
