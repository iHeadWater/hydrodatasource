"""
Author: liutiaxqabs 1498093445@qq.com
Date: 2024-04-19 13:58:31
LastEditors: liutiaxqabs 1498093445@qq.com
LastEditTime: 2024-04-19 14:00:47
FilePath: /liutianxv1/hydrodatasource/hydrodatasource/cleaner/cleaner.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE‘’
cleaner/
│
├── __init__.py
├── cleaner.py          # 包含 Cleaner 基类
├── rainfall_cleaner.py # 包含 RainfallCleaner 类
├── streamflow_cleaner.py # 包含 StreamflowCleaner 类
└── waterlevel_cleaner.py # 包含 WaterlevelCleaner 类
"""


class Cleaner:
    def __init__(self, filepath):
        self.filepath = filepath

    def read_data(self):
        # 实现读取数据的方法
        pass

    def save_data(self, data, output_path):
        # 实现保存数据的方法
        pass

    def anomaly_process(self, methods=None):
        if methods is None:
            methods = []
        # 如果有特定流程，可以在这里添加
        pass
