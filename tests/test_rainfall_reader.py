"""
Author: liutiaxqabs 1498093445@qq.com
Date: 2025-01-15 10:42:21
LastEditors: Wenyu Ouyang
LastEditTime: 2025-01-15 14:07:11
FilePath: \hydrodatasource\tests\test_rainfall_reader.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
"""

import pytest
import pandas as pd
from hydrodatasource.reader.rainfall_reader import RainfallReader


@pytest.fixture
def station_rainfall_reader(tmpdir):
    # Create a temporary directory for the test data
    data_folder = tmpdir.mkdir("data")
    # Create a sample station info file
    pptn_info_file = data_folder.join("pptn_info.csv")
    pptn_info_file.write("STCD\n0001\n0002\n")
    # Create sample rainfall data files for stations
    for station_id in ["0001", "0002"]:
        station_dir = data_folder.mkdir(station_id)
        station_data_file = station_dir.join(f"pp_CHN_songliao_{station_id}.csv")
        station_data_file.write(
            "TM,STCD,DRP\n2023-01-01,0001,10.0\n2023-01-02,0001,12.0\n"
        )
    return RainfallReader(data_folder)


def test_read_1station_rainfall(station_rainfall_reader, abnormal=True):
    # Test reading rainfall data for station 0001
    station_id = "0001"
    df = station_rainfall_reader.read_1station_rainfall(station_id)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert list(df.columns) == ["STCD", "DRP"]
    assert df.index.name == "TM"
    assert df.loc["2023-01-01", "DRP"] == 10.0
    assert df.loc["2023-01-02", "DRP"] == 12.0

    # Test reading rainfall data for station 0002
    station_id = "0002"
    df = station_rainfall_reader.read_1station_rainfall(station_id)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert list(df.columns) == ["STCD", "DRP"]
    assert df.index.name == "TM"
    assert df.loc["2023-01-01", "DRP"] == 10.0
    assert df.loc["2023-01-02", "DRP"] == 12.0
