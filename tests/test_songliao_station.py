import glob
import os.path
import shutil

import geopandas as gpd
import pandas as pd
import psycopg2
from hydrodatasource.cleaner.streamflow_cleaner import StreamflowCleaner

import hydrodatasource.configs.config as conf


def get_specific_shp(gdf, stcd_column='stcd', target_stcd=None):
    """
    从GeoDataFrame中提取指定站点代码的多边形外边界坐标。

    参数:
    gdf: GeoDataFrame对象，包含站点数据和几何信息。
    stcd_column: 字符串，表示站点代码的列名。
    target_stcd: 字符串，表示需要提取坐标的目标站点代码。

    返回:
    一个字典，键为站点代码，值为对应多边形外边界坐标的二维数组。
    """
    coordinates_dict = {}
    # 筛选出特定的站点代码
    if target_stcd is not None:
        try:
            gdf = gdf[gdf[stcd_column].isin(target_stcd)]
        except:
            gdf = gdf[gdf[stcd_column] == target_stcd]

    for index, row in gdf.iterrows():
        # 获取站点代码
        stcd = row[stcd_column]
        # 获取多边形外边界的坐标
        try:
            if row['geometry'].geom_type == 'Polygon':
                exterior_coords = [(lon, lat) for lon, lat in row['geometry'].exterior.coords]
            else:
                # 使用 geoms 属性访问 MultiPolygon 中的所有多边形
                multi_exterior_coords = []
                for polygon in row['geometry'].geoms:
                    # 遍历每个多边形的外边界坐标
                    multi_exterior_coords.extend([(lon, lat) for lon, lat in polygon.exterior.coords])
                exterior_coords = multi_exterior_coords
            # 存储坐标
            coordinates_dict[stcd] = exterior_coords
        except:
            pass

    return coordinates_dict


def get_geom_text(points):
    """
    取多边形边界值文本，逆时针。
    返回值可以给postgis提供polygon
    SELECT ST_GeomFromText('LINESTRING(-124.95 49.95,-124.95 25.05,-79.95 25.05,-66.05 25.75,-66.05 47.95,-66.75 49.55,-66.85 49.65,-87.35 49.95,-124.95 49.95)');

    构造四至方法：
    SELECT ST_MakePolygon(ST_GeomFromText('LINESTRING(115 38,115 50,135 50,135 38 ,115 38)'));
    取polygon四至的方法：
    select st_xmin(ST_MakePolygon(ST_GeomFromText('LINESTRING(-78.35 36.65,-77.85 36.65,-77.85 36.75,-78.05 36.949999999999996,-78.25 37.05,-78.35 37.05,-78.35 36.65)')))
    union all
    select st_ymin(ST_MakePolygon(ST_GeomFromText('LINESTRING(-78.35 36.65,-77.85 36.65,-77.85 36.75,-78.05 36.949999999999996,-78.25 37.05,-78.35 37.05,-78.35 36.65)')))
    union all
    select st_xmax(ST_MakePolygon(ST_GeomFromText('LINESTRING(-78.35 36.65,-77.85 36.65,-77.85 36.75,-78.05 36.949999999999996,-78.25 37.05,-78.35 37.05,-78.35 36.65)')))
    union all
    select st_ymax(ST_MakePolygon(ST_GeomFromText('LINESTRING(-78.35 36.65,-77.85 36.65,-77.85 36.75,-78.05 36.949999999999996,-78.25 37.05,-78.35 37.05,-78.35 36.65)')))

    select
        *
    from
        t_locations tl
    where
        ST_Contains(
        ST_MakePolygon(ST_GeomFromText('LINESTRING ( 121.312350 30.971457 , 121.156783 31.092221 , 121.353250 31.278195 , 121.509125 31.157431 , 121.312350 30.971457 ) ')) ,
        st_point(longitude,latitude)
        )
    :param points:
    :return:
    """
    # 取边界点
    # logger.warning(points[hull.vertices, 0])
    # logger.warning(points[hull.vertices, 1])
    GeomText = ''
    start_node = ''
    # 逆时针取定点，并组合闭环
    i = 0
    for (lon, lat) in points:
        coordinate = str(lon) + ' ' + str(lat)
        GeomText = GeomText + coordinate + ','
        if i == 0:
            start_node = coordinate
        i += 1
    GeomText = GeomText + start_node
    return GeomText


def get_geom_info(coordinates):
    """
    根据坐标取坐标外沿等
    :param coordinates:
    :return:
        bbox   四至
        points  所有坐标点
        edge_points 外沿
    """
    lat_min = None
    lat_max = None
    lon_min = None
    lon_max = None
    points = []
    for coordinate in coordinates:
        points.append(coordinate)
        if None in [lat_min, lat_max, lon_min, lon_max]:
            lon_max = coordinate[0]
            lon_min = coordinate[0]
            lat_min = coordinate[1]
            lat_max = coordinate[1]
        else:
            if lon_max > coordinate[0]:
                lon_max = coordinate[0]
            if lon_min < coordinate[0]:
                lon_min = coordinate[0]
            if lat_min > coordinate[1]:
                lat_min = coordinate[1]
            if lat_max < coordinate[1]:
                lat_max = coordinate[1]
    lon_list = [lon_min, lon_max]
    lat_list = [lat_min, lat_max]
    return min(lon_list), max(lon_list), min(lat_list), max(lat_list), points


def get_station_by_shp(all_watershed, basin_code):
    """
    获取shp内的包含的测站列表
    """
    # logger.error("开始获取shape")
    # all_watershed = gpd.read_file(conf.FS.open("s3://basins-origin/basins_shp.zip"))
    # 从GeoDataFrame中提取指定站点代码的多边形外边界坐标
    coordinates = get_specific_shp(all_watershed, 'BASIN_ID', basin_code)
    lon_min, lon_max, lat_min, lat_max, points = get_geom_info(coordinates[str(basin_code)])
    # logger.error("获取shape结束")
    geomText = get_geom_text(points)
    sql = """
        select
            stcd,
            rname as stcdname,
            lon,
            lat,
            sttype,
            newid,
            "source"
        from
            v_stbprp_b
        where
            ST_Contains(
                ST_MakePolygon(ST_GeomFromText('LINESTRING(geotext)')),
                v_stbprp_b.geom
            );
    """
    sql = sql.replace('geotext', geomText)
    # logger.info(sql)
    df = pd.read_sql(
        sql,
        psycopg2.connect(
            'postgresql://postgres:water@10.55.0.102:5432/water'
        )
    )
    return df


def test_songliao_station():
    all_watershed = gpd.read_file(conf.FS.open("s3://basins-origin/basins_shp.zip"))
    chn_gage_id = [gage_id.split('/')[-1].split('.')[0].split('_')[1] for gage_id in glob.glob('/ftproot/basins-interim/timeseries/3h/*.csv')
                   if 'songliao' in gage_id]
    for gage_id in chn_gage_id:
        try:
            df = get_station_by_shp(all_watershed, gage_id)
        except KeyError:
            # 11205200, 11200400, 11300400, 11007700, 10811000, .etc
            gage_id = str(gage_id)
            df = pd.read_sql(f"SELECT * FROM v_stbprp_b where stcd = '{gage_id}'", psycopg2.connect(
                'postgresql://postgres:water@10.55.0.102:5432/water'
            ))
            sttype_ = df['sttype'].values[0]
            if (sttype_ == 'ZZ') | (sttype_ == 'ZQ'):
                df = pd.read_sql(f"SELECT * FROM st_river_r where stcd = '{gage_id}'", psycopg2.connect(
                    'postgresql://postgres:water@10.55.0.102:5432/water'
                ))
            elif sttype_ == 'RR':
                df = pd.read_sql(f"SELECT * FROM st_rsvr_r where stcd = '{gage_id}'", psycopg2.connect(
                    'postgresql://postgres:water@10.55.0.102:5432/water'
                ))
            elif sttype_ == 'PP':
                df = pd.read_sql(f"SELECT * FROM st_pptn_r where stcd = '{gage_id}'", psycopg2.connect(
                    'postgresql://postgres:water@10.55.0.102:5432/water'
                ))
            else:
                print('What the F**k')
        df.to_csv(os.path.join('sl_stcds', gage_id + '.csv'))


def test_read_songliao_station_data():
    import numpy as np
    gage_mdata = glob.glob('sl_stcds/*.csv')
    type_dict = {'RR': 'st_rsvr_r', 'ZZ': 'st_river_r', 'ZQ': 'st_river_r', 'PP': 'st_pptn_r'}
    for gage_id in gage_mdata:
        df = pd.read_csv(gage_id)
        if 'sttype' in df.columns:
            stcd_series = df['stcd'][(df['sttype'].str.contains('|'.join(['RR', 'ZZ', 'ZQ', 'PP']))) & (~df['sttype'].isna())]
            for stcd in stcd_series.tolist():
                csv_path = os.path.join('sl_stcd_datas', 'songliao_'+str(stcd) + '.csv')
                if os.path.exists(csv_path):
                    continue
                else:
                    table_name = type_dict[df['sttype'][df['stcd']==stcd].values[0]]
                    data_df = pd.read_sql(f"SELECT * FROM {table_name} where stcd = '{stcd}'", psycopg2.connect(
                    'postgresql://postgres:water@10.55.0.102:5432/water'), parse_dates=['tm'])
                    if (np.min(data_df['tm']) < pd.to_datetime('2023-06-01')) & (len(data_df) > 0):
                        data_df.to_csv(csv_path)
        else:
            shutil.copy(gage_id, os.path.join('sl_stcd_datas', gage_id.split('/')[-1]))


def test_process_ema():
    csv_files = glob.glob('sl_stcd_datas/*.csv')
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        if 'drp' not in df.columns:
            save_path = csv_file.replace('sl_stcd_datas', 'sl_stcd_datas_filtered')
            if not os.path.exists(save_path):
                cleaner = StreamflowCleaner(csv_file)
                # methods默认可以联合调用，也可以单独调用。大多数情况下，默认调用moving_average
                cleaner.anomaly_process(methods=["EMA"])
                cleaner.processed_df.to_csv(save_path,index=False)
