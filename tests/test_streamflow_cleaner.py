'''
Author: liutiaxqabs 1498093445@qq.com
Date: 2024-04-22 13:38:07
LastEditors: liutiaxqabs 1498093445@qq.com
LastEditTime: 2024-06-13 16:18:40
FilePath: /hydrodatasource/tests/test_streamflow_cleaner.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

import pytest
from hydrodatasource.cleaner.streamflow_cleaner import StreamflowCleaner, StreamflowBacktrack  # 确保引入你的类
import pandas as pd
import matplotlib.pyplot as plt

def test_anomaly_process():
    # 测试径流数据处理功能，单独处理csv文件，修改该过程可实现文件夹批处理多个文件
    cleaner = StreamflowCleaner("/ftproot/tests_stations_anomaly_detection/streamflow_cleaner/21401550.csv")
    # methods默认可以联合调用，也可以单独调用。大多数情况下，默认调用moving_average
    methods = ["EMA"]
    cleaner.anomaly_process(methods)
    print(cleaner.origin_df)
    print(cleaner.processed_df)
    cleaner.processed_df.to_csv("/ftproot/tests_stations_anomaly_detection/streamflow_cleaner/21401550.csv",index=False)

def test_process_backtrack():
    # 测试径流数据反推处理功能
    cleaner = StreamflowBacktrack("/ftproot/tests_stations_anomaly_detection/streamflow_backtrack/","/ftproot/tests_stations_anomaly_detection/streamflow_backtrack/")
    cleaner.process_backtrack()
