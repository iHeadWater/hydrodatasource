"""
Author: liutiaxqabs 1498093445@qq.com
Date: 2024-04-22 13:38:07
LastEditors: Wenyu Ouyang
LastEditTime: 2025-01-07 14:26:40
FilePath: \hydrodatasource\tests\test_streamflow_cleaner.py
Description: Test funcs for streamflow data cleaning
"""

import os
import glob
from tqdm import tqdm
import pandas as pd
import pytest
from hydrodatasource.cleaner.streamflow_cleaner import StreamflowBacktrack

from hydrodatasource.cleaner.streamflow_cleaner import (
    StreamflowCleaner,
)


@pytest.fixture
def setup_test_environment(tmpdir):
    # Create a temporary directory for test files
    input_dir = tmpdir.mkdir("input")
    output_dir = tmpdir.mkdir("output")

    # Create a sample CSV file with test data
    test_data = {
        "TM": pd.date_range(start="2023-01-01", periods=10, freq="D"),
        "RZ": [100, 150, 300, 350, 400, 450, 500, 550, 600, 650],
        "W": [10, 15, 20, 25, 30, 35, 40, 45, 50, 55],
    }
    test_df = pd.DataFrame(test_data)
    input_file = os.path.join(input_dir, "test_data.csv")
    test_df.to_csv(input_file, index=False)

    return input_file, output_dir


def test_clean_w(setup_test_environment):
    input_file, output_dir = setup_test_environment

    # Initialize the StreamflowBacktrack object
    backtrack = StreamflowBacktrack(data_folder="", output_folder="")

    # Call the clean_w method
    cleaned_file = backtrack.clean_w(input_file, output_dir)

    # Check if the cleaned file exists
    assert os.path.exists(cleaned_file), "Cleaned file was not created."

    # Load the cleaned data
    cleaned_data = pd.read_csv(cleaned_file)

    # Check if the NaN values were set correctly
    assert cleaned_data["RZ"].isna().sum() > 0, "NaN values were not set correctly."

    # Check if the cleaned data file has the expected columns
    expected_columns = ["TM", "RZ", "W", "diff_prev", "diff_next", "set_nan"]
    assert all(
        column in cleaned_data.columns for column in expected_columns
    ), "Cleaned data does not have the expected columns."

    # Check if the plot file was created
    plot_file = os.path.join(output_dir, "rsvr_w_clean.png")
    assert os.path.exists(plot_file), "Plot file was not created."


def test_clean_w_no_nan(setup_test_environment):
    input_file, output_dir = setup_test_environment

    # Modify the test data to have no NaN values
    test_data = {
        "TM": pd.date_range(start="2023-01-01", periods=10, freq="D"),
        "RZ": [100, 150, 200, 250, 300, 350, 400, 450, 500, 550],
        "W": [10, 15, 20, 25, 30, 35, 40, 45, 50, 55],
    }
    test_df = pd.DataFrame(test_data)
    test_df.to_csv(input_file, index=False)

    # Initialize the StreamflowBacktrack object
    backtrack = StreamflowBacktrack(data_folder="", output_folder="")

    # Call the clean_w method
    cleaned_file = backtrack.clean_w(input_file, output_dir)

    # Load the cleaned data
    cleaned_data = pd.read_csv(cleaned_file)

    # Check if no NaN values were set
    assert cleaned_data["RZ"].isna().sum() == 0, "Unexpected NaN values were set."

    # Check if the cleaned data file has the expected columns
    expected_columns = ["TM", "RZ", "W", "diff_prev", "diff_next", "set_nan"]
    assert all(
        column in cleaned_data.columns for column in expected_columns
    ), "Cleaned data does not have the expected columns."

    # Check if the plot file was created
    plot_file = os.path.join(output_dir, "rsvr_w_clean.png")
    assert os.path.exists(plot_file), "Plot file was not created."


def test_back_calculation(setup_test_environment):
    input_file, output_dir = setup_test_environment

    # Modify the test data to include necessary columns for back_calculation
    test_data = {
        "TM": pd.date_range(start="2023-01-01", periods=10, freq="D"),
        "OTQ": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
        "W": [100, 150, 200, 250, 300, 350, 400, 450, 500, 550],
        "STCD": [1] * 10,
        "RZ": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "BLRZ": [0] * 10,
        "RWCHRCD": [0] * 10,
        "RWPTN": [0] * 10,
        "INQDR": [0] * 10,
        "MSQMT": [0] * 10,
    }
    test_df = pd.DataFrame(test_data)
    test_df.to_csv(input_file, index=False)

    # Initialize the StreamflowBacktrack object
    backtrack = StreamflowBacktrack(data_folder="", output_folder="")

    # Call the back_calculation method
    back_calc_file = backtrack.back_calculation(input_file, "test_data.csv", output_dir)

    # Check if the back calculation file exists
    assert os.path.exists(back_calc_file), "Back calculation file was not created."

    # Load the back calculation data
    back_calc_data = pd.read_csv(back_calc_file)

    # Check if the back calculation data file has the expected columns
    expected_columns = [
        "TM",
        "RZ",
        "INQ",
        "W",
        "BLRZ",
        "OTQ",
        "RWCHRCD",
        "RWPTN",
        "INQDR",
        "MSQMT",
    ]
    assert all(
        column in back_calc_data.columns for column in expected_columns
    ), "Back calculation data does not have the expected columns."

    # Check if the INQ values were calculated correctly -- the first value is nan, so we skip it
    assert (
        back_calc_data["INQ"][1:].notna().all()
    ), "INQ values were not calculated correctly."


def test_anomaly_process():
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
