"""
Author: liutiaxqabs 1498093445@qq.com
Date: 2024-05-15 10:26:29
LastEditors: Wenyu Ouyang
LastEditTime: 2025-10-31 11:03:18
FilePath: \hydrodatasource\tests\test_dmca_esr.py
Description: TODO: This test file need to be refactored
"""

import os
import numpy as np
from pint import UnitRegistry

from hydrodataset import CamelsUs

from hydrodatasource.configs.config import SETTING
from hydrodatasource.processor.dmca_esr import *
from hydroutils import hydro_units


def test_rainfall_runoff_event_identify():
    camels = CamelsUs(os.path.join(SETTING["local_data_path"]["datasets-origin"]))
    gage_ids = camels.read_object_ids()
    ureg = UnitRegistry()

    rain = camels.read_ts_xrdataset(
        gage_ids[:1], ["1980-01-01", "2015-01-01"], var_lst=["precipitation"]
    )
    flow = camels.read_ts_xrdataset(
        gage_ids[:1], ["1980-01-01", "2015-01-01"], var_lst=["streamflow"]
    )
    # trans unit to mm/day
    basin_area = camels.read_area(gage_ids[:1])
    r_mmd = hydro_units.streamflow_unit_conv(
        flow, basin_area, target_unit="mm/d", source_unit="m^3/s"
    )
    flow_threshold = hydro_units.streamflow_unit_conv(
        np.array([100]) * ureg.m**3 / ureg.s,
        basin_area.isel(basin=0).to_array().to_numpy() * ureg.km**2,
        # for the flow threshold, we use mm/h as the unit
        target_unit="mm/h",
        source_unit="m^3/s",
    )
    flood_events = rainfall_runoff_event_identify(
        rain["precipitation"].isel(basin=0).to_series(),
        r_mmd["streamflow"].isel(basin=0).to_series(),
        flow_threshold=flow_threshold[0],
    )
    assert flood_events["BEGINNING_RAIN"].shape[0] == flood_events["END_RAIN"].shape[0]
