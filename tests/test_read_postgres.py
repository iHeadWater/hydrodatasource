"""
Author: Wenyu Ouyang
Date: 2024-08-09 13:19:39
LastEditTime: 2025-03-19 19:03:29
LastEditors: Wenyu Ouyang
Description: 
FilePath: \hydrodatasource\tests\test_read_postgres.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""
from hydrodatasource.reader.postgres import read_forcing_dataframe, read_plcd
import hydrodatasource.configs.config as hdscc
import pandas as pd


def test_read_gpm():
    data = read_forcing_dataframe("gpm_tp", "21401550", ["2024-05-19 06:00:00", None])
    print(data[["predictdate", "tp"]])


def test_read_gfs_tp():
    data = read_forcing_dataframe(
        "gfs_tp", "21401550", ["2024-05-19 06:00:00", "2024-05-19 07:00:00"]
    )
    print(data[["forecastdatetime", "tp"]])


def test_read_smap():
    data = read_forcing_dataframe(
        "smap_sm_surface", "21401550", ["2024-05-02 00:00:00", "2024-05-03 00:00:00"]
    )
    print(data[["predictdate", "sm_surface"]])


def test_read_gfs_sm():
    data = read_forcing_dataframe(
        "gfs_soilw", "21401550", ["2024-06-01 00:00:00", "2024-06-02 00:00:00"]
    )
    print(data[["forecastdatetime", "sm_surface"]])


def test_read_sl_pg():
    import geopandas as gpd

    # 获取water数据库下所有表
    all_tables = pd.read_sql(
        "SELECT table_name FROM information_schema.tables WHERE table_type = 'BASE TABLE' "
        "AND table_catalog='water'",
        hdscc.PS,
    )
    # 获取基础表信息
    stbprp_df = pd.read_sql("select * FROM ST_STBPRP_B", hdscc.PS)
    print(all_tables)
    print(stbprp_df)
    """
    stnames = ['猴山', '石门', '土门子', '三湾', '铁甲', '太平湾水电站', '水丰', '双岭', '南城子', '榛子岭', '丰满', '杨木',
               '云峰水电站', '红石', '白山', '双沟', '小山', '松山', '向海', '老龙口']
    stnames_df = stbprp_df[['STCD', 'STNAME', 'LGTD', 'LTTD']][stbprp_df['STNM'].str.contains('|'.join(stnames))]
    # 若点处于多个流域的出口，取级别最低的那一个
    geo_column = gpd.points_from_xy(stnames_df['LGTD'], stnames_df['LTTD'])
    sta_gdf = gpd.GeoDataFrame(stnames_df).set_geometry(geo_column)
    """


def test_read_plcd():
    basin_code = "10513040"
    data = read_plcd(basin_code)
    print(data)
    print(type(data))
