"""
Author: Wenyu Ouyang
Date: 2023-10-31 09:26:31
LastEditTime: 2025-01-09 14:57:53
LastEditors: Wenyu Ouyang
Description: Reading cleaned reservoir inflow data
FilePath: \hydrodatasource\hydrodatasource\reader\rsvr_inflow_reader.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

from hydrodatasource.reader.data_source import HydroData


class RsvrInflow(HydroData):
    def __init__(self, data_dir):
        self.data_source_dir = data_dir
        self.data_source_description = self.set_data_source_describe()
    
    def set_data_source_describe(self):
        return "Reservoir inflow data from the cleaned data source"
