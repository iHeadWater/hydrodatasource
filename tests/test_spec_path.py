from hydrodata.reader import access_fs


def test_read_spec():
    # access_fs.spec_path("st_rain_c.csv")
    access_fs.spec_path('grids-interim/86_21401550/gpm_gfs.nc', head='minio')
