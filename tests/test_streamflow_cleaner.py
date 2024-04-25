'''
Author: liutiaxqabs 1498093445@qq.com
Date: 2024-04-22 13:38:07
LastEditors: liutiaxqabs 1498093445@qq.com
LastEditTime: 2024-04-24 13:32:34
FilePath: /hydrodatasource/tests/test_streamflow_cleaner.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import pytest
from hydrodatasource.cleaner.streamflow_cleaner import StreamflowCleaner  # 确保引入你的类
import pandas as pd
import matplotlib.pyplot as plt

def test_anomaly_process():
    # 测试径流数据处理功能
    cleaner = StreamflowCleaner("/home/liutianxv1/径流sampledatatest.csv", cwt_row=10)
    # methods默认可以联合调用，也可以单独调用。大多数情况下，默认调用moving_average
    methods = ['wavelet']
    cleaner.anomaly_process(methods)
    print(cleaner.origin_df)
    print(cleaner.processed_df)
    cleaner.processed_df.to_csv("/home/liutianxv1/径流sampledatatest.csv")
