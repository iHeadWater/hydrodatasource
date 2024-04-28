from hydrodatasource.reader.postgres import read_data


def test_read_rainfall_data():
    data = read_data(stcd='10800800', datatype='rain', start_time='2013-08-16T08:00:00', end_time='2013-08-17T08:00:00')
    print(data)


def test_read_streamflow_data():
    data = read_data(stcd='21401550', datatype='streamflow', start_time='2020-07-01T00:00:00',
                     end_time='2020-07-31T23:00:00')
    print(data)
