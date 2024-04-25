'''
Author: liutiaxqabs 1498093445@qq.com
Date: 2024-04-22 18:02:00
LastEditors: liutiaxqabs 1498093445@qq.com
LastEditTime: 2024-04-24 13:33:01
FilePath: /hydrodatasource/tests/test_rainfall_cleaner.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

import pytest
from hydrodatasource.cleaner.rainfall_cleaner import RainfallCleaner
import pandas as pd
import matplotlib.pyplot as plt


def test_anomaly_process():
    # 测试降雨数据处理功能
    cleaner = RainfallCleaner(
        "/home/liutianxv1/降雨sampledatatest.csv", grad_max=200, extr_max=200
    )
    # methods默认可以联合调用，也可以单独调用。大多数情况下，默认调用detect_sum
    methods = ["extreme", "gradient", "detect_sum"]
    cleaner.anomaly_process(methods)

    print(cleaner.origin_df)
    print(cleaner.processed_df)
    # cleaner.processed_df.to_csv("/home/liutianxv1/降雨sampledatatest.csv")
    cleaner.temporal_list.to_csv("/home/liutianxv1/降雨temporal_list.csv")
