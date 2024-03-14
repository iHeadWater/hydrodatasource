from hydrodata.reader import access_fs


def test_read_spec():
    # access_fs.spec_path("st_rain_c.csv")
    mean_forcing_nc = access_fs.spec_path(
        "basins-origin/hour_data/1h/mean_data/mean_data_forcing/mean_forcing_CHN_21401550.nc",
        head="minio",
    )
    print(mean_forcing_nc)
