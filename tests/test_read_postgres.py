from hydrodatasource.reader.postgres import read_forcing, get_forcing_dataframe


def test_read_gpm():
    data = read_forcing("gpm", "21401550", "2024-05-24 00:00:00")
    print(data)


def test_read_gfs_tp():
    data = read_forcing("gfs_tp", "21401550", "2024-05-24 00:00:00")
    print(data)


def test_read_smap():
    data = read_forcing("smap", "21401550", "2024-05-20 14:00:00")
    print(data)


def test_read_gfs_sm():
    data = read_forcing("gfs_soil", "21401550", "2024-05-20 19:00:00")
    print(data)


def test_get_dataframe_gpm():
    data = get_forcing_dataframe("gpm", "21401550", "2024-05-24 00:00:00")
    print(data)
