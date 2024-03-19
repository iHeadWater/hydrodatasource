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


def test_read_BA():
    basin = HydroBasins(data_path="./")  # 该路径只是为了实例化该类，测试时可随意指定
    attr = basin.read_BA_xrdataset(
        gage_id_lst=["21401550"], var_lst=["all"], path="basins-origin/attributes.nc"
    )
    print(attr.compute())



def test_read_pp_stations_shp():
    """
    # 读取zip中的shpfiles文件
    pp_stations = gpd.read_file(
        conf.FS.open(
            "s3://stations-origin/stations_list/pp_stations.zip"
        )
    )
    
    """
    # 读取csv文件
    pp_stations = access_fs.spec_path(
        "stations-origin/stations_list/pp_stations.csv",
        head = 'minio',
        )
    print(pp_stations)