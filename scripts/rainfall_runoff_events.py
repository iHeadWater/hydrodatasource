"""
Author: liutiaxqabs 1498093445@qq.com
Date: 2024-03-16 15:55:22
LastEditTime: 2025-01-14 14:39:09
LastEditors: Wenyu Ouyang
Description: see rainfall runoff events
FilePath: \hydrodatasource\scripts\rainfall_runoff_events.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import os
from pathlib import Path
import sys


sys.path.append(os.path.dirname(Path(os.path.abspath(__file__)).parent))
from const4scripts import DATASET_DIR, RESULT_DIR
from hydrodatasource.reader.data_source import SelfMadeHydroDataset
from hydrodatasource.processor.dmca_esr import get_rr_events, plot_rr_events

datasource_dir = os.path.join(DATASET_DIR, "FDSources")
datasource = SelfMadeHydroDataset(datasource_dir)
BASIN_ID = ["songliao_21401550"]
T_RANGE = ["2000-10-01", "2023-12-31"]
rain_flow = datasource.read_ts_xrdataset(
    BASIN_ID, T_RANGE, ["total_precipitation_hourly", "streamflow"]
)["1D"]
basin_area = datasource.read_area(BASIN_ID)
rr_events = get_rr_events(
    rain_flow["total_precipitation_hourly"],
    rain_flow["streamflow"],
    basin_area,
    max_window=7,
    max_flow_min=[10],
)
save_dir = os.path.join(RESULT_DIR, "rr_events")
plot_rr_events(
    rr_events,
    rain_flow["total_precipitation_hourly"],
    rain_flow["streamflow"],
    save_dir=save_dir,
)
