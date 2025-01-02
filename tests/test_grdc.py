"""
Author: Wenyu Ouyang
Date: 2025-01-02 17:31:19
LastEditTime: 2025-01-02 19:43:57
LastEditors: Wenyu Ouyang
Description: 
FilePath: \hydrodatasource\tests\test_grdc.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import os
import collections
import pytest
from hydrodatasource.reader.grdc import Grdc
import pandas as pd
from hydrodatasource.reader.grdc import _grdc_read


@pytest.fixture
def grdc_instance(tmp_path):
    data_path = tmp_path / "data"
    data_path.mkdir()
    for continent in [
        "africa",
        "asia",
        "south_america",
        "north_america",
        "south_west_pacific",
        "europe",
    ]:
        continent_path = data_path / continent
        continent_path.mkdir()
        (continent_path / "daily").mkdir()
        (continent_path / "monthly").mkdir()
    (data_path / "GRDC_Watersheds").mkdir()
    return Grdc(data_path)


def test_set_data_source_describe(grdc_instance):
    description = grdc_instance.set_data_source_describe()
    data_root_dir = grdc_instance.data_source_dir

    expected_description = collections.OrderedDict(
        DATA_DIR=data_root_dir,
        CONTINENT_DATA={
            "africa": {
                "daily": os.path.join(data_root_dir, "africa", "daily"),
                "monthly": os.path.join(data_root_dir, "africa", "monthly"),
            },
            "asia": {
                "daily": os.path.join(data_root_dir, "asia", "daily"),
                "monthly": os.path.join(data_root_dir, "asia", "monthly"),
            },
            "south_america": {
                "daily": os.path.join(data_root_dir, "south_america", "daily"),
                "monthly": os.path.join(data_root_dir, "south_america", "monthly"),
            },
            "north_america": {
                "daily": os.path.join(data_root_dir, "north_america", "daily"),
                "monthly": os.path.join(data_root_dir, "north_america", "monthly"),
            },
            "south_west_pacific": {
                "daily": os.path.join(data_root_dir, "south_west_pacific", "daily"),
                "monthly": os.path.join(data_root_dir, "south_west_pacific", "monthly"),
            },
            "europe": {
                "daily": os.path.join(data_root_dir, "europe", "daily"),
                "monthly": os.path.join(data_root_dir, "europe", "monthly"),
            },
        },
        SHAPE_DIR=os.path.join(data_root_dir, "GRDC_Watersheds"),
    )

    assert description == expected_description


@pytest.fixture
def grdc_station_file(tmp_path):
    file_content = """# Title:                 GRDC STATION DATA FILE
#                        --------------
# Format:                DOS-ASCII
# Field delimiter:       ;
# missing values are indicated by -999.000
#
# file generation date:  2024-05-29
#
# GRDC-No.:              4101200
# River:                 WULIK RIVER
# Station:               BELOW TUTAK CREEK NEAR KIVALINA, AK
# Country:               US
# Latitude (DD):       67.8754
# Longitude (DD):      -163.6774
# Catchment area (km?:      1825.95
# Altitude (m ASL):        53.34
# Next downstream station:      -
# Remarks:               
# Owner of original data: United States of America - US Geological Survey (USGS)
#************************************************************
#
# Data Set Content:      MEAN DAILY DISCHARGE (Q)
#                        --------------------
# Unit of measure:                  m?s
# Time series:           1984-09 - 2022-10
# No. of years:          39
# Last update:           2023-05-09
#
# Table Header:
#     YYYY-MM-DD - Date
#     hh:mm      - Time
#     Value   - original (provided) data
#************************************************************
#
# Data lines: 13909
# DATA
YYYY-MM-DD;hh:mm; Value
1984-09-11;--:--;     15.574
1984-09-12;--:--;     15.404
1984-09-13;--:--;     15.065
1984-09-14;--:--;     14.895
    """
    file_path = tmp_path / "4101200_Q_Day.Cmd.txt"
    file_path.write_text(file_content, encoding="utf-8")
    return file_path


def test_grdc_read(grdc_station_file):
    start_date = "1984-09-11"
    end_date = "1984-09-14"
    column = "Value"

    metadata, df = _grdc_read(grdc_station_file, start_date, end_date, column)

    expected_metadata = {
        "grdc_file_name": str(grdc_station_file),
        "id_from_grdc": 4101200,
        "file_generation_date": "2024-05-29",
        "river_name": "WULIK RIVER",
        "station_name": "BELOW TUTAK CREEK NEAR KIVALINA, AK",
        "country_code": "US",
        "grdc_latitude_in_arc_degree": 67.8754,
        "grdc_longitude_in_arc_degree": -163.6774,
        "grdc_catchment_area_in_km2": 1825.95,
        "altitude_masl": 53.34,
        "dataSetContent": "MEAN DAILY DISCHARGE (Q)",
        "units": "m?s",
        "time_series": "1984-09 - 2022-10",
        "no_of_years": 39,
        "last_update": "2023-05-09",
        "nrMeasurements": 13909,
    }

    expected_df = pd.DataFrame(
        {column: [15.574, 15.404, 15.065, 14.895]},
        index=pd.date_range(start=start_date, end=end_date),
    )
    expected_df.index.rename("time", inplace=True)

    assert metadata == expected_metadata
    pd.testing.assert_frame_equal(df, expected_df)
