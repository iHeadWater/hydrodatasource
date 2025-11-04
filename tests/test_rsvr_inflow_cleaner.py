"""
Author: liutiaxqabs 1498093445@qq.com
Date: 2024-04-22 13:38:07
LastEditors: Wenyu Ouyang
LastEditTime: 2025-11-04 10:06:49
FilePath: \hydrodatasource\tests\test_rsvr_inflow_cleaner.py
Description: Test funcs for streamflow data cleaning
"""

import os
import numpy as np
import pandas as pd
import pytest

# Set matplotlib to non-interactive backend for testing
import matplotlib

matplotlib.use("Agg")

from hydrodatasource.cleaner.rsvr_inflow_cleaner import (
    ReservoirInflowBacktrack,
    linear_interpolate_wthresh,
)


@pytest.fixture
def setup_test_environment(tmpdir):
    # Create a temporary directory for test files
    input_dir = tmpdir.mkdir("input")
    output_dir = tmpdir.mkdir("output")

    # Create a sample CSV file with test data
    # Note: RZ should be within [DDZ=50, DSFLZ=120] and W should be within [DDCP=10, TTCP=100]
    # Include some out-of-range values to test the cleaning functionality
    test_data = {
        "TM": pd.date_range(
            start="2023-01-01", periods=100, freq="h"
        ),  # Ensure enough rows, use 'h' instead of deprecated 'H'
        "RZ": [
            55,
            60,
            65,
            70,
            75,
            80,
            150,
            90,
            95,
            100,
        ]  # 150 is out of range [50, 120]
        * 10,  # Mix of valid and invalid values
        "W": [15, 20, 25, 30, 35, 40, 120, 50, 55, 60]  # 120 is out of range [10, 100]
        * 10,  # Mix of valid and invalid values
        "MSQMT": [0] * 100,
        "STCD": ["001"] * 100,
        "OTQ": [5, 10, 15, 20, 25, 30, -5, 40, 45, 50] * 10,  # -5 is negative (invalid)
        "BLRZ": [0] * 100,
        "RWCHRCD": [0] * 100,
        "INQ": [10, 12.5, np.nan, np.nan, 20, 25, 30, np.nan, 35, 40] * 10,
        "RWPTN": [0] * 100,
        "INQDR": [0] * 100,
    }
    test_df = pd.DataFrame(test_data)
    input_file = os.path.join(input_dir, "001_rsvr_data.csv")
    test_df.to_csv(input_file, index=False)

    # Create dummy reservoir info files
    rsvr_idname_file = os.path.join(input_dir, "rsvr_stcd_stnm.xlsx")
    rsvr_charact_waterlevel_file = os.path.join(
        input_dir, "rsvr_charact_waterlevel.csv"
    )
    pd.DataFrame({"STCD": ["001"], "STNM": ["Reservoir1"]}).to_excel(
        rsvr_idname_file, index=False
    )
    pd.DataFrame(
        {
            "STCD": ["001"],
            "NORMZ": [100],
            "DDZ": [50],
            "DSFLZ": [120],
            "DDCP": [10],
            "TTCP": [100],
        }
    ).to_csv(rsvr_charact_waterlevel_file, index=False)

    return input_file, output_dir, input_dir


def test_clean_w(setup_test_environment):
    input_file, output_dir, input_dir = setup_test_environment

    # Initialize the ReservoirInflowBacktrack object
    backtrack = ReservoirInflowBacktrack(
        data_folder=input_dir, output_folder=output_dir
    )

    # Call the clean_w method
    cleaned_file = backtrack.clean_w("001", input_file, output_dir)

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
    input_file, output_dir, input_dir = setup_test_environment

    # Modify the test data to have no NaN values
    # All values should be within valid ranges
    # Need at least 72 rows for validation to pass
    test_data = {
        "TM": pd.date_range(start="2023-01-01", periods=80, freq="h"),
        "RZ": [55, 60, 65, 70, 75, 80, 85, 90, 95, 100] * 8,  # All within [50, 120]
        "W": [15, 20, 25, 30, 35, 40, 45, 50, 55, 60] * 8,  # All within [10, 100]
        "MSQMT": [0] * 80,
        "STCD": ["001"] * 80,
        "OTQ": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50] * 8,  # All positive
        "BLRZ": [0] * 80,
        "RWCHRCD": [0] * 80,
        "INQ": [10, 12.5, 15, 17.5, 20, 25, 30, 32, 35, 40] * 8,  # No NaN values
        "RWPTN": [0] * 80,
        "INQDR": [0] * 80,
    }
    test_df = pd.DataFrame(test_data)
    test_df.to_csv(input_file, index=False)

    # Initialize the ReservoirInflowBacktrack object
    backtrack = ReservoirInflowBacktrack(
        data_folder=input_dir, output_folder=output_dir
    )

    # Call the clean_w method
    cleaned_file = backtrack.clean_w("001", input_file, output_dir)

    # Load the cleaned data
    cleaned_data = pd.read_csv(cleaned_file)

    # Check if no NaN values were set (all data should be valid)
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
    input_file, output_dir, input_dir = setup_test_environment

    # Modify the test data to include necessary columns for back_calculation
    # Need at least 72 rows for validation and all required columns
    test_data = {
        "TM": pd.date_range(start="2023-01-01", periods=80, freq="h"),
        "OTQ": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50] * 8,
        "W": [15, 20, 25, 30, 35, 40, 45, 50, 55, 60] * 8,
        "STCD": ["001"] * 80,
        "RZ": [55, 60, 65, 70, 75, 80, 85, 90, 95, 100] * 8,
        "INQ": [np.nan]
        * 80,  # INQ column is required even if it's NaN for back calculation
        "BLRZ": [0] * 80,
        "RWCHRCD": [0] * 80,
        "RWPTN": [0] * 80,
        "INQDR": [0] * 80,
        "MSQMT": [0] * 80,
    }
    test_df = pd.DataFrame(test_data)
    test_df.to_csv(input_file, index=False)

    # Initialize the ReservoirInflowBacktrack object
    backtrack = ReservoirInflowBacktrack(
        data_folder=input_dir, output_folder=output_dir
    )

    # Call the back_calculation method
    back_calc_file = backtrack.back_calculation(
        "001", input_file, input_file, output_dir
    )

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


def test_delete_negative_inq(setup_test_environment):
    input_file, output_dir, input_dir = setup_test_environment

    # Modify the test data to include necessary columns for delete_nan_inq
    # Need at least 72 rows for validation
    test_data = {
        "TM": pd.date_range(start="2023-01-01", periods=80, freq="h"),
        "INQ": [10, -5, 15, -10, 20, -15, 25, -20, 30, -25] * 8,
        "RZ": [55, 60, 65, 70, 75, 80, 85, 90, 95, 100] * 8,
        "W": [15, 20, 25, 30, 35, 40, 45, 50, 55, 60] * 8,
        "OTQ": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50] * 8,
        "STCD": ["001"] * 80,
        "BLRZ": [0] * 80,
        "RWCHRCD": [0] * 80,
        "RWPTN": [0] * 80,
        "INQDR": [0] * 80,
        "MSQMT": [0] * 80,
    }
    test_df = pd.DataFrame(test_data)
    test_df.to_csv(input_file, index=False)

    # Initialize the ReservoirInflowBacktrack object
    backtrack = ReservoirInflowBacktrack(
        data_folder=input_dir, output_folder=output_dir
    )

    # Call the delete_nan_inq method
    cleaned_file = backtrack.delete_negative_inq(
        "001",
        input_file,
        input_file,  # Use the same file path as input for original data
        output_dir,
        negative_deal_window=5,
        negative_deal_stride=3,
    )

    # Check if the cleaned file exists
    assert os.path.exists(cleaned_file), "Cleaned file was not created."

    # Load the cleaned data
    cleaned_data = pd.read_csv(cleaned_file)

    # Check if the cleaned data file has the expected columns
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
        column in cleaned_data.columns for column in expected_columns
    ), "Cleaned data does not have the expected columns."

    # Check if the INQ values were adjusted correctly, as stride exist, cannot deal with all data
    assert (
        cleaned_data["INQ"][:-1] >= 0
    ).all(), "INQ values were not adjusted correctly."

    # Check if the sum of INQ values is balanced (use relaxed tolerance for rounding errors)
    original_sum = test_df["INQ"].sum()
    cleaned_sum = cleaned_data["INQ"].sum()
    # Relaxed tolerance because the cleaning process involves sliding window adjustments
    assert (
        abs(original_sum - cleaned_sum) < 1.0
    ), "INQ values are not balanced correctly."


def test_linear_interpolate(setup_test_environment):
    input_file, output_dir, input_dir = setup_test_environment

    # Modify the test data to include NaN values for interpolation
    test_data = {
        "TM": pd.date_range(start="2023-01-01", periods=10, freq="D"),
        "INQ": [10, 12.5, np.nan, np.nan, 20, 25, 30, np.nan, 35, 40],
        "RZ": [100, 150, 200, 250, 300, 350, 400, 450, 500, 550],
        "W": [10, 15, 20, 25, 30, 35, 40, 45, 50, 55],
    }
    test_df = pd.DataFrame(test_data)
    test_df.to_csv(input_file, index=False)

    # Load the test data
    df = pd.read_csv(input_file)

    # Call the linear_interpolate function
    interpolated_df = linear_interpolate_wthresh(df, column="INQ", threshold=3)

    # Check if the NaN values were interpolated correctly
    expected_inq = [10, 12.5, 15, 17.5, 20, 25, 30, 32.5, 35, 40]
    assert np.allclose(
        interpolated_df["INQ"], expected_inq, equal_nan=True
    ), "INQ values were not interpolated correctly."

    # Check if the DataFrame still has the expected columns
    expected_columns = ["TM", "INQ", "RZ", "W"]
    assert all(
        column in interpolated_df.columns for column in expected_columns
    ), "Interpolated data does not have the expected columns."

    # Check if the interpolation respects the threshold
    test_data = {
        "TM": pd.date_range(start="2023-01-01", periods=10, freq="D"),
        "INQ": [10, np.nan, np.nan, np.nan, np.nan, 25, 30, np.nan, 35, 40],
        "RZ": [100, 150, 200, 250, 300, 350, 400, 450, 500, 550],
        "W": [10, 15, 20, 25, 30, 35, 40, 45, 50, 55],
    }
    test_df = pd.DataFrame(test_data)
    test_df.to_csv(input_file, index=False)

    # Load the test data
    df = pd.read_csv(input_file)

    # Call the linear_interpolate function with a threshold of 3
    interpolated_df = linear_interpolate_wthresh(df, column="INQ", threshold=3)

    # Check if the NaN values were interpolated correctly
    expected_inq = [10, np.nan, np.nan, np.nan, np.nan, 25, 30, 32.5, 35, 40]
    assert np.allclose(
        interpolated_df["INQ"], expected_inq, equal_nan=True
    ), "INQ values were not interpolated correctly with threshold."

    # Check if the DataFrame still has the expected columns
    assert all(
        column in interpolated_df.columns for column in expected_columns
    ), "Interpolated data does not have the expected columns."


def test_insert_inq(setup_test_environment):
    input_file, output_dir, input_dir = setup_test_environment

    # Modify the test data to include necessary columns for insert_inq
    test_data = {
        "TM": pd.date_range(start="2023-01-01", periods=100, freq="0.5h"),
        "INQ": [10, np.nan, 15, np.nan, 20, np.nan, 25, np.nan, 30, np.nan] * 10,
        "RZ": [100, 150, 200, 250, 300, 350, 400, 450, 500, 550] * 10,
        "W": [10, 15, 20, 25, 30, 35, 40, 45, 50, 55] * 10,
        "OTQ": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50] * 10,
        "STCD": ["001"] * 100,
        "BLRZ": [0] * 100,
        "RWCHRCD": [0] * 100,
        "RWPTN": [0] * 100,
        "INQDR": [0] * 100,
        "MSQMT": [0] * 100,
    }
    test_df = pd.DataFrame(test_data)
    test_df.to_csv(input_file, index=False)

    # Initialize the ReservoirInflowBacktrack object
    backtrack = ReservoirInflowBacktrack(
        data_folder=input_dir, output_folder=output_dir
    )

    # Call the insert_inq method
    result_file = backtrack.insert_inq("001", input_file, input_file, output_dir)

    # Check if the result file exists
    assert os.path.exists(result_file), "Result file was not created."

    # Load the result data
    result_data = pd.read_csv(result_file)

    # Check if the result data file has the expected columns
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
        column in result_data.columns for column in expected_columns
    ), "Result data does not have the expected columns."

    # Check if the INQ values were interpolated correctly
    assert (
        result_data["INQ"].isna().sum() == 0
    ), "INQ values were not interpolated correctly."

    # Check if the INQ values are non-negative
    assert (result_data["INQ"] >= 0).all(), "INQ values are not non-negative."

    # Check if the STCD values are consistent
    assert result_data["STCD"].nunique() == 1, "STCD values are not consistent."
