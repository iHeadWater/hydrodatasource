from hydrodata.reader import access_fs
import hydrodata.configs.config as conf
import geopandas as gpd
import pandas as pd


from hydrodata.reader.data_source import HydroBasins


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
    return zz_stations_gdf,pp_stations_gdf,zq_stations_gdf


def test_read_stations_list():
    # 读取csv文件
    zz_stations_df = pd.read_csv(
        "stations-origin/stations_list/zz_stations.csv",
        storage_options=conf.MINIO_PARAM,index_col= False
    )
    print("zz_stations 站点列表如下:")
    print(zz_stations_df)
    pp_stations_df = pd.read_csv(
        "stations-origin/stations_list/pp_stations.csv",
        storage_options=conf.MINIO_PARAM,index_col= False
    )
    print("zz_stations 站点列表如下:")
    print(pp_stations_df)
    zq_stations_df = pd.read_csv(
        "stations-origin/stations_list/zq_stations.csv",
        storage_options=conf.MINIO_PARAM,index_col= False
    )
    print("zz_stations 站点列表如下:")
    print(zq_stations_df)
    return zz_stations_df,pp_stations_df,zq_stations_df

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
