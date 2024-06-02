from hydrodatasource.reader.postgres import (
    read_forcing_dataframe,
)


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
