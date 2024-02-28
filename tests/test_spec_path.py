from hydrodata.reader import access_fs


def test_read_spec():
    # access_fs.spec_path("st_rain_c.csv")
    # gpm_gfs_nc = access_fs.spec_path('grids-interim/86_21401550/gpm_gfs.nc', head='minio')
    print_json = access_fs.spec_path('test/geodata/era5_land/era5_land_.json', head='minio', need_cache=False)
