from hydrodatasource.configs.config import PS
from datetime import datetime


def read_forcing(var_type, basin, start_time):
    if start_time is None:
        raise ValueError("start_time cannot be None")

    table_name = {
        "gpm": "t_gpm_pre_data",
        "gfs_tp": "t_gfs_tp_pre_data",
        "smap": "t_smap_pre_data",
        "gfs_sm": "t_gfs_soil_pre_data",
    }

    # 连接数据库
    ps = PS.cursor()

    if var_type not in table_name:
        raise ValueError("var_type must be one of 'gpm', 'gfs_tp', 'smap', 'gfs_sm'")

    if var_type in ["gpm", "smap"]:
        sql = f"""
        SELECT * FROM {table_name[var_type]}
        WHERE predictdate >= %s AND basincode = %s
        """
    elif var_type in ["gfs_tp", "gfs_sm"]:
        sql = f"""
        SELECT * FROM {table_name[var_type]}
        WHERE forecastdatetime >= %s AND basin_code = %s
        """

    # 执行查询
    PS.autocommit = True
    ps.execute(sql, (start_time, basin))
    results = ps.fetchall()

    # 关闭游标和连接
    ps.close()

    return results
