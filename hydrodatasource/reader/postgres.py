from hydrodatasource.configs.config import PS
from datetime import datetime

def read_data(stcd, datatype, start_time, end_time):
    ps = PS.cursor()
    dt_start = datetime.strptime(start_time, "%Y-%m-%dT%H:%M:%S")
    dt_end = datetime.strptime(end_time, "%Y-%m-%dT%H:%M:%S")
    dt_start_time = dt_start.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
    dt_end_time = dt_end.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
    if datatype == 'rain':
        table = 'st_pptn_r'
        columns = "tm, intv, drp"
    elif datatype == 'streamflow':
        table = 'st_rsvr_r'
        columns = "tm, inq"
    else:
        raise ValueError("datatype must be 'rain' or 'streamflow'")
    sql = f"SELECT {columns} FROM {table} where stcd = %s and tm between %s and %s"
    PS.autocommit = True
    ps.execute(sql, (stcd, dt_start_time, dt_end_time))
    return ps.fetchall()