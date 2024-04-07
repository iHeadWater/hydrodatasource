import geopandas as gpd
import pandas as pd
import pytest
import xarray as xr
import hydrodatasource.configs.config as conf
from hydrodatasource.reader import access_fs
from hydrodatasource.cleaner import rain_anomaly 
from hydrodatasource.reader.data_source import HydroBasins
def test_read_spec():
    # access_fs.spec_path("st_rain_c.csv")
    mean_forcing_nc = access_fs.spec_path(
        "basins-origin/hour_data/1h/mean_data/mean_data_forcing/mean_forcing_CHN_21401550.nc",
        head="minio",
    )
    print(mean_forcing_nc)


def test_read_shp():
    watershed = gpd.read_file(
        conf.FS.open(
            "s3://basins-origin/basin_shapefiles/basin_USA_camels_01411300.zip"
        )
    )
    print(watershed)

    all_watershed = gpd.read_file(conf.FS.open("s3://basins-origin/basins_shp.zip"))
    print(all_watershed)


def test_read_BA():
    basin = HydroBasins(data_path="./")  # 该路径只是为了实例化该类，测试时可随意指定
    attr = basin.read_BA_xrdataset(
        gage_id_lst=["21401550"], var_lst=["all"], path="basins-origin/attributes.nc"
    )
    print(attr.compute())

    all_attr = access_fs.spec_path(
        "basins-origin/attributes.nc",
        head="minio",
    )
    print(all_attr.compute())


def test_read_zz_stations_ts():
    # 读取csv文件
    zz_stations = access_fs.spec_path(
        "stations-origin/zz_stations/hour_data/1h/zz_CHN_songliao_10800100.csv",
        head="minio",
    )
    print(zz_stations)


def test_read_stations_shp():
    # 读取zip中的shpfiles文件
    zz_stations_gdf = gpd.read_file(
        conf.FS.open("s3://stations-origin/stations_list/zz_stations.zip")
    )
    print("zz_stations 站点列表如下:")
    print(zz_stations_gdf)
    # 读取zip中的shpfiles文件
    pp_stations_gdf = gpd.read_file(
        conf.FS.open("s3://stations-origin/stations_list/pp_stations.zip")
    )
    print("pp_stations 站点列表如下:")
    print(pp_stations_gdf)
    # 读取zip中的shpfiles文件
    zq_stations_gdf = gpd.read_file(
        conf.FS.open("s3://stations-origin/stations_list/zq_stations.zip")
    )
    print("zq_stations 站点列表如下:")
    print(zq_stations_gdf)
    return zz_stations_gdf, pp_stations_gdf, zq_stations_gdf


def test_read_stations_list():
    # 读取csv文件
    zz_stations_df = pd.read_csv(
        "s3://stations-origin/stations_list/zz_stations.csv",
        storage_options=conf.MINIO_PARAM,
        index_col=False,
    )
    print("zz_stations 站点列表如下:")
    print(zz_stations_df)
    pp_stations_df = pd.read_csv(
        "s3://stations-origin/stations_list/pp_stations.csv",
        storage_options=conf.MINIO_PARAM,
        index_col=False,
    )
    print("pp_stations 站点列表如下:")
    print(pp_stations_df)
    zq_stations_df = pd.read_csv(
        "s3://stations-origin/stations_list/zq_stations.csv",
        storage_options=conf.MINIO_PARAM,
        index_col=False,
    )
    print("zq_stations 站点列表如下:")
    print(zq_stations_df)
    return zz_stations_df, pp_stations_df, zq_stations_df


def test_read_zqstations_ts():
    test_csv = pd.read_csv(
        "s3://stations-origin/zq_stations/zq_CHN_songliao_10310500.csv",
        storage_options=conf.MINIO_PARAM,
    )
    return test_csv


def test_read_reservoirs_info():
    dams_gdf = gpd.read_file(conf.FS.open("s3://reservoirs-origin/dams.zip"))
    rsvrs_gdf = gpd.read_file(conf.FS.open("s3://reservoirs-origin/rsvrs_shp.zip"))
    return dams_gdf, rsvrs_gdf


def test_read_river_network():
    test_gdf = gpd.read_file(conf.FS.open("s3://basins-origin/HydroRIVERS_v10_shp.zip"))
    return test_gdf


def test_read_rsvr_ts():
    test_rsvr_df = pd.read_csv(
        "s3://reservoirs-origin/rr_stations/zq_CHN_songliao_10310500.csv",
        storage_options=conf.MINIO_PARAM,
    )
    return test_rsvr_df


def test_read_pp():
    pp_df = pd.read_csv(
        "s3://stations-origin/pp_stations/hour_data/1h/pp_CHN_songliao_10951870.csv",
        storage_options=conf.MINIO_PARAM,
    )
    return pp_df


def test_read_zz():
    zz_df = pd.read_csv(
        "s3://stations-origin/zz_stations/hour_data/1h/zz_CHN_dalianxiaoku_21302120.csv",
        storage_options=conf.MINIO_PARAM,
    )
    return zz_df


def test_read_zq():
    zq_df = pd.read_csv(
        "s3://stations-origin/zq_stations/hour_data/1h/zq_USA_usgs_01181000.csv",
        storage_options=conf.MINIO_PARAM,
    )
    return zq_df


def test_df2ds():
    zq_df = pd.read_csv(
        "s3://stations-origin/zq_stations/hour_data/1h/zq_USA_usgs_01181000.csv",
        storage_options=conf.MINIO_PARAM,
    )
    # 两种写法都可以，但索引列名不同
    zq_ds = xr.Dataset().from_dataframe(zq_df)  # Coordinates: index
    # zq_dss = xr.Dataset(zq_df) # Coordinates: dim_0
    return zq_ds

def list_csv_files(bucket_name, prefix=''):
    """
    列出指定S3桶中所有CSV文件的路径。
    """
    # 使用FS的glob方法查找匹配的文件
    path = f'{bucket_name}/{prefix}' if prefix else f'{bucket_name}'
    files = conf.FS.glob(f'{path}/*.csv')
    # s3fs glob方法返回的路径是完整的s3路径
    return files

def read_process_store_csv(src_bucket, dest_bucket, prefix=''):
    """
    读取源桶中的CSV文件，处理它们，并将结果存储到目标桶。
    """
    csv_files = list_csv_files(src_bucket, prefix)
    total_files = len(csv_files)  # 总文件数
    print(f"总共发现 {total_files} 个CSV文件。")
    
    for file_path in csv_files:
        # 从S3读取CSV到DataFrame
        with conf.FS.open(file_path, mode='rb') as f:
            df = pd.read_csv(f,index_col = False)
        
        # 处理DataFrame 单步调用
        df_processed = rain_anomaly.rainfall_format_normalization(df)
        print(df_processed.head())
        
        # 从原始文件路径中提取文件名
        file_name = file_path.split('/')[-1]
        # 构建新的目标文件路径
        dest_file_path = f"{dest_bucket}/{file_name}"
        
        # 将处理后的DataFrame写入新的目标文件路径
        with conf.FS.open(dest_file_path, 'w') as f:
            df.to_csv(f, index=False)


def test_read_folder():
    src_bucket = 's3://stations-interim/pp_stations/hour_data/1h'
    dest_bucket = 's3://stations-interim/pp_stations/hour_data/1h'
    read_process_store_csv(src_bucket, dest_bucket)