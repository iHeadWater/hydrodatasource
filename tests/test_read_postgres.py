from hydrodatasource.reader.postgres import read_forcing


def test_read_gpm():
    data = read_forcing("gpm", "21401550", "2024-05-24 00:00:00")
    print(data)


def test_read_gfs_tp():
    data = read_forcing("gfs_tp", "21401550", "2024-05-24 00:00:00")
    print(data)


def test_read_smap():
    data = read_forcing("smap", "21401550", "2024-05-24 00:00:00")
    print(data)


def test_read_gfs_sm():
    data = read_forcing("gfs_sm", "21401550", "2024-05-24 00:00:00")
    print(data)
