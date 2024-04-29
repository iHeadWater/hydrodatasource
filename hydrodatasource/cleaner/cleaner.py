'''
Author: liutiaxqabs 1498093445@qq.com
Date: 2024-04-19 13:58:31
LastEditors: liutiaxqabs 1498093445@qq.com
LastEditTime: 2024-04-26 17:09:33
FilePath: /hydrodatasource/hydrodatasource/cleaner/cleaner.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
"""
cleaner/
│
├── __init__.py
├── cleaner.py          # 包含 Cleaner 基类
├── rainfall_cleaner.py # 包含 RainfallCleaner 类
├── streamflow_cleaner.py # 包含 StreamflowCleaner 类
└── waterlevel_cleaner.py # 包含 WaterlevelCleaner 类
"""

import xarray as xr
import pandas as pd
import numpy as np


class Cleaner:
    def __init__(self, data_path, *args, **kwargs):
        self.data_path = data_path
        self.origin_df = None
        self.processed_df = None
        self.read_data()

    def read_data(self):
        # 读取数据并存储在origin_df中
        self.origin_df = pd.read_csv(self.data_path, dtype={"STCD": str}, index_col=False)
        self.processed_df = self.origin_df.copy()

    def save_data(self, data, output_path):
        # 保存数据到CSV
        data.to_csv(output_path, index=False)

    def anomaly_process(self, methods=None):
        if methods is None:
            methods = []
        # 如果有特定流程，可以在这里添加
        pass
