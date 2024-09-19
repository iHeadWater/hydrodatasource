import pandas as pd
import geopandas as gpd
import sqlalchemy as sqa
from hydrodatasource.cleaner.streamflow_cleaner import StreamflowCleaner, StreamflowBacktrack  # 确保引入你的类
# network_gdf=gpd.read_file('/home/wangyang1/songliao_cut_single.shp')
node_shp = gpd.read_file("/home/wangyang1/463_nodes_sl/463_nodes_sl.shp")

def test_dload_463_nodes():
    # 创建数据库连接
    engine = sqa.create_engine("mssql+pymssql://sa:water^2021@10.10.50.189:1433/rtdb")

    stcd_str = node_shp['STCD'].astype(str).to_list()

    # 构建查询语句，从 ST_RSVR_R 表中获取与清河和柴河水库对应的数据
    for stcd in stcd_str:
        query_rsvr = f"SELECT * FROM ST_RSVR_R WHERE STCD == '{stcd}'"

    # 执行查询并将结果保存为 DataFrame
        ST_RSVR_ZZ = pd.read_sql(query_rsvr, engine)
        ST_RSVR_ZZ.to_csv(f'sl_stcds/{stcd}.csv')

def test_anomaly_process():
    # 测试径流数据处理功能，单独处理csv文件，修改该过程可实现文件夹批处理多个文件
    cleaner = StreamflowCleaner("/ftproot/tests_stations_anomaly_detection/streamflow_cleaner/21401550.csv")
    # methods默认可以联合调用，也可以单独调用。大多数情况下，默认调用moving_average
    methods = ["EMA"]
    cleaner.anomaly_process(methods)
    print(cleaner.origin_df)
    print(cleaner.processed_df)
    cleaner.processed_df.to_csv("/ftproot/tests_stations_anomaly_detection/streamflow_cleaner/21401550.csv",index=False)
