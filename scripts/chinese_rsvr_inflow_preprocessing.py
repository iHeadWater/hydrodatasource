"""
Author: Wenyu Ouyang
Date: 2025-01-06 20:34:34
LastEditTime: 2025-01-07 20:50:48
LastEditors: Wenyu Ouyang
Description: script for chinese streamflow preprocessing
FilePath: \hydrodatasource\scripts\chinese_rsvr_inflow_preprocessing.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import glob
import os
import sys
from pathlib import Path

from tqdm import tqdm

from hydrodatasource.cleaner.streamflow_cleaner import StreamflowCleaner


sys.path.append(os.path.dirname(Path(os.path.abspath(__file__)).parent))
from const4scripts import RESULT_DIR, DATASET_DIR
from hydrodatasource.cleaner.rsvr_inflow_cleaner import (
    ReservoirInflowBacktrack,
)


# 测试径流数据反推处理功能
original_reservoir_data_dir = os.path.join(DATASET_DIR, "数据库原始流量")
tmp_dir = os.path.join(RESULT_DIR, "反推流量")
rsvr_inflow_backtrack = ReservoirInflowBacktrack(
    data_folder=original_reservoir_data_dir,
    output_folder=tmp_dir,
)
rsvr_inflow_backtrack.process_backtrack()


# 测试径流数据处理功能，单独处理csv文件，修改该过程可实现文件夹批处理多个文件
cleaner = StreamflowCleaner(
    "/ftproot/tests_stations_anomaly_detection/streamflow_cleaner/21312150.csv",
    window_size=7,
)
# methods默认可以联合调用，也可以单独调用。大多数情况下，默认调用moving_average
methods = ["EMA"]
cleaner.anomaly_process(methods)
print(cleaner.origin_df)
print(cleaner.processed_df)
cleaner.processed_df.to_csv(
    "/ftproot/tests_stations_anomaly_detection/streamflow_cleaner/21312150.csv",
    index=False,
)


def test_anomaly_process_folder():
    input_folder = "/home/liutianxv1/EMA评估/日尺度松辽数据源/"
    output_folder = "/home/liutianxv1/EMA评估/日尺度松辽数据源/"

    # 获取输入文件夹中所有CSV文件的路径
    csv_files = glob.glob(os.path.join(input_folder, "*.csv"))
    print(csv_files)

    for csv_file in tqdm(csv_files):
        try:
            # 读取并处理每个CSV文件
            cleaner = StreamflowCleaner(
                csv_file, window_size=7, cutoff_frequency=0.1, iterations=2, cwt_row=1
            )  # 国内window_size=14,stride=1,cutoff_frequency=0.035,time_step=1.0,iterations=3,sampling_rate=1.0,order=5,cwt_row=2,
            methods = ["ewma"]
            cleaner.anomaly_process(methods)

            # 确定输出文件路径
            output_file = os.path.join(output_folder, os.path.basename(csv_file))

            # 保存处理后的数据
            cleaner.processed_df.to_csv(output_file, index=False)

            print(f"Processed {csv_file} and saved to {output_file}")
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
