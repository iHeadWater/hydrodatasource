import geopandas as gpd
import pandas as pd
import pytest
import numpy as np
import xarray as xr
import hydrodatasource.configs.config as conf
from hydrodatasource.reader import access_fs


def test_read_zz_stations_ts():
    # 读取csv文件
    zz_stations = access_fs.spec_path(
        "stations-origin/zz_stations/hour_data/1h/zz_CHN_songliao_10800100.csv",
        head="minio",
    )
    print(zz_stations)
