'''
Author: liutiaxqabs 1498093445@qq.com
Date: 2024-04-22 13:38:07
LastEditors: liutiaxqabs 1498093445@qq.com
LastEditTime: 2024-10-11 10:24:32
FilePath: /hydrodatasource/tests/test_streamflow_cleaner.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

import pytest
from hydrodatasource.cleaner.streamflow_cleaner import StreamflowCleaner, StreamflowBacktrack  # 确保引入你的类
import pandas as pd
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
import os

def test_anomaly_process():
    # 测试径流数据处理功能，单独处理csv文件，修改该过程可实现文件夹批处理多个文件
    cleaner = StreamflowCleaner("/ftproot/tests_stations_anomaly_detection/streamflow_cleaner/21312150.csv",window_size=7)
    # methods默认可以联合调用，也可以单独调用。大多数情况下，默认调用moving_average
    methods = ["EMA"]
    cleaner.anomaly_process(methods)
    print(cleaner.origin_df)
    print(cleaner.processed_df)
    cleaner.processed_df.to_csv("/ftproot/tests_stations_anomaly_detection/streamflow_cleaner/21312150.csv",index=False)

def test_anomaly_process_folder():
    input_folder = "/home/liutianxv1/EMA评估/日尺度松辽数据源/"
    output_folder = "/home/liutianxv1/EMA评估/日尺度松辽数据源/"

    # 获取输入文件夹中所有CSV文件的路径
    csv_files = glob.glob(os.path.join(input_folder, "*.csv"))
    print(csv_files)

    for csv_file in tqdm(csv_files):
        try:
            # 读取并处理每个CSV文件
            cleaner = StreamflowCleaner(csv_file,window_size=7,cutoff_frequency=0.1,iterations=2,cwt_row=1)#国内window_size=14,stride=1,cutoff_frequency=0.035,time_step=1.0,iterations=3,sampling_rate=1.0,order=5,cwt_row=2,
            methods = ["ewma"]
            cleaner.anomaly_process(methods)
            
            # 确定输出文件路径
            output_file = os.path.join(output_folder, os.path.basename(csv_file))

            # 保存处理后的数据
            cleaner.processed_df.to_csv(output_file, index=False)

            print(f"Processed {csv_file} and saved to {output_file}")
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")

def test_process_backtrack():
    # 测试径流数据反推处理功能
    cleaner = StreamflowBacktrack(data_folder="/home/liutianxv1/EWMA重新制作/IntermediateData/数据库原始流量",output_folder="/home/liutianxv1/EWMA重新制作/IntermediateData/反推流量")
    cleaner.process_backtrack()

def test_delete_nan_inq():
    # 测试径流数据反推处理功能
    cleaner = StreamflowBacktrack("/ftproot/tests_stations_anomaly_detection/streamflow_backtrack/","/ftproot/tests_stations_anomaly_detection/streamflow_backtrack/")
    cleaner.delete_nan_inq(data_path="/ftproot/tests_stations_anomaly_detection/streamflow_backtrack/zq_CHN_songliao_21401550.csv",
                           file="zq_CHN_songliao_21401550.csv",
                           output_folder="/ftproot/tests_stations_anomaly_detection/streamflow_backtrack/")
    