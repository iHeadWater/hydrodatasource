from hydrodatasource.configs.config import PS
from datetime import datetime
import pandas as pd


def read_forcing(var_type, basin, start_time):
    if start_time is None:
        raise ValueError("start_time cannot be None")

    table_name = {
        "gpm_tp": "t_gpm_pre_data",
        "gfs_tp": "t_gfs_tp_pre_data",
        "smap": "t_smap_pre_data",
        "gfs_soil": "t_gfs_soil_pre_data",
    }

    # 连接数据库
    ps = PS.cursor()

    if var_type not in table_name:
        raise ValueError(
            "var_type must be one of 'gpm_tp', 'gfs_tp', 'smap', 'gfs_soil'"
        )

    if var_type == "gpm_tp":
        sql = f"""
        SELECT 
            basincode, 
            predictdate, 
            data ->> 'tp' AS tp,
            data ->> 'raster_area' AS raster_area,
            data ->> 'intersection_area' AS intersection_area
        FROM (
            SELECT 
                basincode, 
                predictdate, 
                jsonb_array_elements(data) AS data
            FROM {table_name[var_type]}
        ) {table_name[var_type]}
        WHERE predictdate >= %s AND basincode = %s
        """
    elif var_type == "smap":
        sql = f"""
        SELECT 
            basincode, 
            predictdate, 
            data ->> 'sm_surface' AS sm_surface,
            data ->> 'raster_area' AS raster_area,
            data ->> 'intersection_area' AS intersection_area
        FROM (
            SELECT 
                basincode, 
                predictdate, 
                jsonb_array_elements(data) AS data
            FROM {table_name[var_type]}
        ) {table_name[var_type]}
        WHERE predictdate >= %s AND basincode = %s
        """
    elif var_type == "gfs_tp":
        sql = f"""
        select
            basin_code,
            forecastdatetime,
            tp,
            raster_area,
            intersection_area 
        from {table_name[var_type]}
        where forecastdatetime >= %s and basin_code = %s
        """
    elif var_type == "gfs_soil":
        sql = f"""
        select
            basin_code,
            forecastdatetime,
            soilw,
            raster_area,
            intersection_area 
        from {table_name[var_type]}
        where forecastdatetime >= %s and basin_code = %s
        """

    # 执行查询
    PS.autocommit = True
    ps.execute(sql, (start_time, basin))
    results = ps.fetchall()

    # 关闭游标和连接
    ps.close()

    return results


def get_forcing_dataframe(var_type, basin, start_time):
    data = read_forcing(var_type, basin, start_time)
    column_dataname = {
        "gpm_tp": "tp",
        "smap": "sm_surface",
        "gfs_tp": "tp",
        "gfs_soil": "soilw",
    }
    if var_type in ["gpm_tp", "smap"]:
        columns = [
            "basincode",
            "predictdate",
            column_dataname[var_type],
            "raster_area",
            "intersection_area",
        ]
        datetime_column = "predictdate"
    elif var_type in ["gfs_tp", "gfs_soil"]:
        columns = [
            "basin_code",
            "forecastdatetime",
            column_dataname[var_type],
            "raster_area",
            "intersection_area",
        ]
        datetime_column = "forecastdatetime"

    df = pd.DataFrame(data, columns=columns)

    # 转换数据类型
    df[column_dataname[var_type]] = df[column_dataname[var_type]].astype(float)
    df["raster_area"] = df["raster_area"].astype(float)
    df["intersection_area"] = df["intersection_area"].astype(float)

    # 按照时间列排序
    df = df.sort_values(by=datetime_column)

    return df
