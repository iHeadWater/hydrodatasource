from hydrodata.reader import access_fs
import hydrodata.configs.config as conf
import geopandas as gpd


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


def test_read_pp_stations_csv():
    # 读取csv文件
    pp_stations = access_fs.spec_path(
        "stations-origin/stations_list/pp_stations.csv",
        head="minio",
    )
    print(pp_stations)


def test_read_pp_stations_shp():
    # 读取zip中的shpfiles文件
    pp_stations = gpd.read_file(
        conf.FS.open("s3://stations-origin/stations_list/pp_stations.zip")
    )
    print(pp_stations)


def test_read_zz_stations_csv():
    # 读取csv文件
    zz_stations = access_fs.spec_path(
        "stations-origin/stations_list/zz_stations.csv",
        head="minio",
    )
    print(zz_stations)


def test_read_zz_stations_ts():
    # 读取csv文件
    zz_stations = access_fs.spec_path(
        "stations-origin/zz_stations/hour_data/1h/zz_CHN_songliao_10800100.csv",
        head="minio",
    )
    print(zz_stations)


def test_read_zz_stations_shp():
    # 读取zip中的shpfiles文件
    zz_stations = gpd.read_file(
        conf.FS.open("s3://stations-origin/stations_list/zz_stations.zip")
    )
    print(zz_stations)


def test_read_zq_stations_csv():
    # 读取csv文件
    zq_stations = access_fs.spec_path(
        "stations-origin/stations_list/zq_stations.csv",
        head="minio",
    )
    print(zq_stations)


def test_read_zq_stations_shp():
    # 读取zip中的shpfiles文件
    zq_stations = gpd.read_file(
        conf.FS.open("s3://stations-origin/stations_list/zq_stations.zip")
    )
    print(zq_stations)


def test_read_recovered_data():
    import pandas as pd

    test_csv = pd.read_csv(
        "s3://stations-interim/zq_stations/zq_CHN_songliao_10310500.csv",
        storage_options=conf.MINIO_PARAM,
    )
    zq_csv = pd.read_csv(
        "s3://stations-interim/zq_stations.csv", storage_options=conf.MINIO_PARAM
    )
    dams_gdf = gpd.read_file(conf.FS.open("s3://reservoirs-origin/dams.zip"))
    rsvrs_gdf = gpd.read_file(conf.FS.open("s3://reservoirs-origin/rsvrs_shp.zip"))
    rr_df = pd.read_csv(
        "s3://reservoirs-origin/rr_stations/zq_CHN_songliao_10310500.csv",
        storage_options=conf.MINIO_PARAM,
    )
    return test_csv, zq_csv, dams_gdf, rsvrs_gdf, rr_df

def test_read_river_network():
    test_gdf = gpd.read_file(conf.FS.open('s3://basins-origin/HydroRIVERS_v10_shp.zip'))
    return test_gdf

def test_read_rsvr_origin():
    test_rsvr_df = pd.read_csv('s3://reservoirs-origin/rr_stations/zq_CHN_songliao_10310500.csv', storage_options=conf.MINIO_PARAM)
    return test_rsvr_df
