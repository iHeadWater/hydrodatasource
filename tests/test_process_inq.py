"""
Author: Tianxu Liu
Date: 2024-03-16 15:55:22
LastEditTime: 2024-07-06 19:08:11
LastEditors: Wenyu Ouyang
Description: test process inq
FilePath: /hydrodatasource/tests/test_process_inq.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import pytest

from hydrodatasource.cleaner.cleaner import Cleaner


@pytest.fixture
def cleaner():
    return Cleaner(
        file_path="/home/liutianxv/sample_data.csv",
        column_id="ID",
        ID_list=None,
        column_flow="INQ",
        column_time="TM",
        start_time=None,
        end_time=None,
        preprocess=True,
        method="kalman",
        save_path="/home/liutianxv/sample_data.csv",
        plot=True,
        window_size=20,
        cutoff_frequency=0.035,
        iterations=3,
        sampling_rate=1.0,
        time_step=1,
        order=5,
        cwt_row=10,
    )


def test_process_inq(cleaner):
    cleaner.process_inq()
