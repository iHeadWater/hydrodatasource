from hydrodatasource.reader.postgres import (
    read_forcing_dataframe,
)
import hydrodatasource.configs.config as hdscc
import pandas as pd

def test_read_gpm():
    data = read_forcing_dataframe("gpm_tp", "21401550", "2024-05-30 00:00:00")
    print(data)


def test_read_gfs_tp():
    data = read_forcing_dataframe("gfs_tp", "21401550", "2024-05-20 00:00:00")
    print(data)
    data.to_csv("test.csv")


def test_read_smap():
    data = read_forcing_dataframe("smap", "21401550", "2024-05-20 14:00:00")
    print(data)


def test_read_gfs_sm():
    data = read_forcing_dataframe("gfs_soil", "21401550", "2024-05-20 19:00:00")
    print(data)


def test_read_sl_pg():
    # 获取water数据库下所有表
    all_tables = pd.read_sql("SELECT table_name FROM information_schema.tables WHERE table_type = 'BASE TABLE' "
                             "AND table_catalog='water'", hdscc.PS)
    # 获取基础表信息
    stbprp_df = pd.read_sql("select * FROM ST_STBPRP_B", hdscc.PS)
    print(all_tables)
    print(stbprp_df)

