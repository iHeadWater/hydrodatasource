'''
Author: liutiaxqabs 1498093445@qq.com
Date: 2024-04-22 18:02:00
LastEditors: liutiaxqabs 1498093445@qq.com
LastEditTime: 2024-04-24 15:49:37
FilePath: /hydrodatasource/tests/test_rainfall_cleaner.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

import pytest
from hydrodatasource.cleaner.waterlevel_cleaner import WaterlevelCleaner
import pandas as pd
import matplotlib.pyplot as plt


def test_anomaly_process():
    # 测试降雨数据处理功能
    cleaner = WaterlevelCleaner(
        "/home/liutianxv1/水位sampledatatest.csv", grad_max=0.5, window_size=168
    )
    # methods默认可以联合调用，也可以单独调用。大多数情况下，默认调用detect_sum
    methods = ["roll","moving_grad"]
    cleaner.anomaly_process(methods)

    print(cleaner.origin_df)
    print(cleaner.processed_df)

    cleaner.processed_df.to_csv("/home/liutianxv1/水位sampledatatest.csv")
