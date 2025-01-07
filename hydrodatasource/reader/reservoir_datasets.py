"""
Author: Wenyu Ouyang
Date: 2023-10-25 17:12:30
LastEditTime: 2025-01-07 21:37:35
LastEditors: Wenyu Ouyang
Description: Reading public reservoir datasets
FilePath: \hydrodatasource\hydrodatasource\reader\reservoir_datasets.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import os
import collections
import geopandas as gpd
from hydrodatasource.reader.data_source import HydroData


class Crd(HydroData):
    """Reading CRD reservoir dataset"""

    def __init__(self, data_path):
        self.data_source_dir = data_path
        self.data_source_description = self.set_data_source_describe()
        self.reservoir_info = self.read_reservoir_info()

    def get_name(self):
        return "CRD"

    def set_data_source_describe(self):
        data_root_dir = self.data_source_dir
        all_rsvrs_dir = os.path.join(data_root_dir, "CRD_v11_all_reservoirs")
        record_rsvrs_dir = os.path.join(data_root_dir, "CRD_v11_record_reservoirs")
        return collections.OrderedDict(
            ALL_RSVRS_SHPFILE=os.path.join(all_rsvrs_dir, "CRD_v11_all_reservoirs.shp"),
            RECORD_RSVRS_SHPFILE=os.path.join(
                record_rsvrs_dir, "CRD_v11_record_reservoirs.shp"
            ),
            # link: https://mp.weixin.qq.com/s/Iy3fFCtGkNc_iu0Ggkh1Lg
        )

    def read_reservoir_info(self):
        all_rsvrs_shpfile = self.data_source_description["ALL_RSVRS_SHPFILE"]
        all_rsvrs_shp = gpd.read_file(all_rsvrs_shpfile)
        return all_rsvrs_shp
