"""
Author: Wenyu Ouyang
Date: 2023-11-01 08:58:50
LastEditTime: 2023-11-01 09:08:27
LastEditors: Wenyu Ouyang
Description: Test funcs for reader.py
FilePath: \hydro_privatedata\tests\test_reader.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""
import os
from hydroprivatedata.config import LOCAL_DATA_PATH
from hydroprivatedata.reader import AOI, StationDataHandler, LocalFileReader


def test_reader_station():
    station_handler = StationDataHandler()
    aoi = AOI(
        "station",
        {"station_id": "2181200", "start_time": "1980-01-01", "end_time": "2001-01-01"},
    )

    local_station_reader = LocalFileReader(station_handler)
    data = local_station_reader.read(
        os.path.join(LOCAL_DATA_PATH, "station.nc"),
        aoi,
    )
    print(data)
