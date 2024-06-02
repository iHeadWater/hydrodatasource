from hydrodatasource.configs.config import SETTING
from datetime import datetime
import pandas as pd
from loguru import logger
from sqlalchemy import create_engine


def read_forcing_dataframe(var_type, basin, start_time):
    if start_time is None:
        raise ValueError("start_time cannot be None")

    table_name = {
        "gpm_tp": "t_gpm_pre_data",
        "gfs_tp": "t_gfs_tp_pre_data",
        "smap": "t_smap_pre_data",
        "gfs_soil": "t_gfs_soil_pre_data",
    }

    column_dataname = {
        "gpm_tp": "tp",
        "smap": "sm_surface",
        "gfs_tp": "tp",
        "gfs_soil": "soilw",
    }

    if var_type in ["gpm_tp", "smap"]:
        columns_mapping = {
            "basincode": "basincode",
            "predictdate": "predictdate",
            column_dataname[var_type]: column_dataname[var_type],
            "raster_area": "raster_area",
            "intersection_area": "intersection_area",
        }
        datetime_column = "predictdate"
    elif var_type in ["gfs_tp", "gfs_soil"]:
        columns_mapping = {
            "basin_code": "basin_code",
            "forecastdatetime": "forecastdatetime",
            column_dataname[var_type]: column_dataname[var_type],
            "raster_area": "raster_area",
            "intersection_area": "intersection_area",
        }
        datetime_column = "forecastdatetime"

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
        WHERE predictdate >= '{start_time}' AND basincode = '{basin}'
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
        WHERE predictdate >= '{start_time}' AND basincode = '{basin}'
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
        where forecastdatetime >= '{start_time}' and basin_code = '{basin}'
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
        where forecastdatetime >= '{start_time}' and basin_code = '{basin}'
        """

    db_username = SETTING["postgres"]["username"]
    db_password = SETTING["postgres"]["password"]
    db_host = SETTING["postgres"]["server_url"]
    db_port = SETTING["postgres"]["port"]
    db_name = SETTING["postgres"]["database"]
    engine = create_engine(
        f"postgresql+psycopg2://{db_username}:{db_password}@{db_host}:{db_port}/{db_name}"
    )
    # 执行查询数据SQL查询
    try:
        df = pd.read_sql(sql, engine)
        df = df.rename(columns=columns_mapping)
        # 转换数据类型
        df[column_dataname[var_type]] = df[column_dataname[var_type]].astype(float)
        df["raster_area"] = df["raster_area"].astype(float)
        df["intersection_area"] = df["intersection_area"].astype(float)
        # 按照时间列排序
        df = df.sort_values(by=datetime_column)
    except Exception as e:
        logger.error(e)
        raise Exception()

    return df
