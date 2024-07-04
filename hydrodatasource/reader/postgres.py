from hydrodatasource.configs.config import SETTING
from datetime import datetime, timedelta
import pandas as pd
from loguru import logger
from sqlalchemy import create_engine


def read_forcing_dataframe(var_type, basin, time_period):
    start_time = time_period[0]
    end_time = time_period[1]
    if start_time is None:
        raise ValueError("the first element of the time_period cannot be None")

    table_name = {
        "gpm_tp": "t_gpm_pre_data",
        "gfs_tp": "t_gfs_tp_pre_data",
        "smap_sm_surface": "t_smap_pre_data",
        "gfs_soilw": "t_gfs_soil_pre_data",
    }

    column_dataname = {
        "gpm_tp": "tp",
        "smap_sm_surface": "sm_surface",
        "gfs_tp": "tp",
        "gfs_soilw": "sm_surface",
    }

    if var_type in ["gpm_tp", "smap_sm_surface"]:
        datetime_column = "predictdate"
    elif var_type in ["gfs_tp", "gfs_soilw"]:
        datetime_column = "forecastdatetime"

    if var_type not in table_name:
        raise ValueError(
            "var_type must be one of 'gpm_tp', 'gfs_tp', 'smap_sm_surface', 'gfs_soilw'"
        )
        
    # 将时间向前调整8小时
    start_time = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
    start_time -= timedelta(hours=8)
    if end_time:
        end_time = datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S')
        end_time -= timedelta(hours=8)
        
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
        WHERE predictdate >= '{start_time}' 
        """
        if end_time is not None:
            sql += f" AND predictdate <= '{end_time}'"
        sql += f" AND basincode = '{basin}'"
    elif var_type == "smap_sm_surface":
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
        WHERE predictdate >= '{start_time}' 
        """
        if end_time is not None:
            sql += f" AND predictdate <= '{end_time}'"
        sql += f" AND basincode = '{basin}'"
    elif var_type == "gfs_tp":
        sql = f"""
        select
            basin_code,
            forecastdatetime,
            tp AS tp,
            raster_area,
            intersection_area 
        from {table_name[var_type]}
        WHERE forecastdatetime >= '{start_time}' 
        """
        if end_time is not None:
            sql += f" AND forecastdatetime <= '{end_time}'"
        sql += f" AND basin_code = '{basin}'"
    elif var_type == "gfs_soilw":
        sql = f"""
        select
            basin_code,
            forecastdatetime,
            soilw AS sm_surface,
            raster_area,
            intersection_area 
        from {table_name[var_type]}
        WHERE forecastdatetime >= '{start_time}' 
        """
        if end_time is not None:
            sql += f" AND forecastdatetime <= '{end_time}'"
        sql += f" AND basin_code = '{basin}'"

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
        # 转换数据类型
        df[column_dataname[var_type]] = df[column_dataname[var_type]].astype(float)
        df["raster_area"] = df["raster_area"].astype(float)
        df["intersection_area"] = df["intersection_area"].astype(float)
        # 按照时间列排序
        df = df.sort_values(by=datetime_column)
        
        df[datetime_column] = pd.to_datetime(df[datetime_column]) + timedelta(hours=8)
    except Exception as e:
        logger.error(e)
        raise Exception() from e

    return df

def read_plcd(basin):
    sql = """
    SELECT
        stcd,
        plcd
    FROM
        t_xaj_parameter
    WHERE
        stcd = %s
    """
    
    db_username = SETTING["postgres"]["username"]
    db_password = SETTING["postgres"]["password"]
    db_host = SETTING["postgres"]["server_url"]
    db_port = SETTING["postgres"]["port"]
    db_name = SETTING["postgres"]["database"]
    
    engine = create_engine(
        f"postgresql+psycopg2://{db_username}:{db_password}@{db_host}:{db_port}/{db_name}"
    )
    
    result = pd.read_sql(sql, engine, params=(basin,))
    
    if result.empty:
        return None
    
    return result.iloc[0]['plcd']