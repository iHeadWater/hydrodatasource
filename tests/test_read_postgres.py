from hydrodatasource.reader.postgres import (
    read_forcing_dataframe,
)


def test_read_gpm():
    data = read_forcing_dataframe("gpm_tp", "21312155", "2024-05-20 00:00:00")
    print(data)


def test_read_gfs_tp():
    data = read_forcing_dataframe("gfs_tp", "21312155", "2024-06-02 00:00:00")
    print(data)


def test_read_smap():
    data = read_forcing_dataframe("smap_sm_surface", "21401550", "2024-05-20 14:00:00")
    print(data)


def test_read_gfs_sm():
    data = read_forcing_dataframe("gfs_soilw", "21401550", "2024-06-03 03:00:00")
    print(data)
