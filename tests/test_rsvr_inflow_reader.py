"""
Author: Wenyu Ouyang
Date: 2023-11-01 08:58:50
LastEditTime: 2025-01-09 18:03:34
LastEditors: Wenyu Ouyang
Description: Test funcs for reader.py
FilePath: \hydrodatasource\tests\test_rsvr_inflow_reader.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import pytest
import pandas as pd
from hydrodatasource.reader.rsvr_inflow_reader import RsvrInflowReader


@pytest.fixture
def rsvr_inflow_reader(tmpdir):
    # Create a temporary directory for the test data
    data_folder = tmpdir.mkdir("data")
    # Create a sample reservoir info file
    rsvr_info_file = data_folder.join("rsvr_info.csv")
    rsvr_info_file.write("STCD\n0001\n0002\n")
    # Create sample inflow data files for reservoirs
    for rsvr_id in ["0001", "0002"]:
        rsvr_dir = data_folder.mkdir(rsvr_id)
        rsvr_data_file = rsvr_dir.join(f"{rsvr_id}_rsvr_data.csv")
        rsvr_data_file.write(
            "TM,STCD,INQ\n2023-01-01,0001,100.0\n2023-01-02,0001,110.0\n"
        )
    return RsvrInflowReader(data_folder)


def test_read_1rsvr_inflow(rsvr_inflow_reader):
    # Test reading inflow data for reservoir 0001
    rsvr_id = "0001"
    df = rsvr_inflow_reader.read_1rsvr_inflow(rsvr_id)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert list(df.columns) == ["STCD", "INQ"]
    assert df.index.name == "TM"
    assert df.loc["2023-01-01", "INQ"] == 100.0
    assert df.loc["2023-01-02", "INQ"] == 110.0

    # Test reading inflow data for reservoir 0002
    rsvr_id = "0002"
    df = rsvr_inflow_reader.read_1rsvr_inflow(rsvr_id)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert list(df.columns) == ["STCD", "INQ"]
    assert df.index.name == "TM"
    assert df.loc["2023-01-01", "INQ"] == 100.0
    assert df.loc["2023-01-02", "INQ"] == 110.0
