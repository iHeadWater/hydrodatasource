import os

import numpy as np
import pandas as pd
from shapely import distance


# 雨量数据异常处理, 参见https://dlut-water.yuque.com/aq3llt/iag1ec/32727520
def get_filter_data_by_time(
    data_path,
    rain_attr="DRP",
    time_attr="TM",
    id_attr="STCD",
    rain_max_hour=200,
    filter_list=None,
):
    """

    Parameters
    ----------
    data_path : _type_
        保存待处理数据文件的文件夹，姑且认为数据文件为“86_单位名称+站号.csv”形式，参考https://dlut-water.yuque.com/kgo8gd/tnld77/pum8d50qrbs1474h
    rain_attr : str, optional
        表格中标示降雨的属性（列名），默认为DRP, by default "DRP"
    time_attr : str, optional
        表格中标示时间的属性（列名），默认为TM, by default "TM"
    id_attr : str, optional
        表格中标示站号的属性（列名），默认为STCD, by default "STCD"
    rain_max_hour : int, optional
        每小时最大降雨量，默认为200，超过这个阈值的排除, by default 200
    filter_list : _type_, optional
        其他预处理过程得到的黑名单，以过滤不可用的站点, by default None

    Returns
    -------
    _type_
        _description_
    """
    if filter_list is None:
        filter_list = []
    time_df_dict = {}
    for dir_name, sub_dir, files in os.walk(data_path):
        for file in files:
            # 目前stcd采用文件名分离出的‘单位名称+站号’，和站点数据表里的站点号未必一致，需要另行考虑
            stcd = (file.split(".")[0]).split("_")[1]
            cached_csv_path = os.path.join(data_path, stcd + ".csv")
            if (stcd not in filter_list) & (~os.path.exists(cached_csv_path)):
                drop_list = []
                csv_path = os.path.join(data_path, file)
                table = pd.read_csv(csv_path, engine="c")
                # 按降雨最大阈值为200和小时雨量一致性过滤索引
                # 有些数据不严格按照小时尺度排列，出于简单可以一概按照小时重采样
                if rain_attr in table.columns:
                    table[time_attr] = pd.to_datetime(
                        table[time_attr], format="%Y-%m-%d %H:%M:%S"
                    )
                    table = table.drop(index=table.index[table[rain_attr].isna()])
                    # 整小时数据，再按小时重采样求和，结果不变
                    table = table.set_index(time_attr).resample("H").sum()
                    cached_time_array = table.index[table[id_attr] != 0].to_numpy()
                    cached_drp_array = table[rain_attr][table[id_attr] != 0].to_numpy()
                    table[time_attr] = np.nan
                    table[rain_attr][cached_time_array] = cached_drp_array
                    table = table.fillna(-1).reset_index()
                    for i in range(len(table[rain_attr])):
                        if table[rain_attr][i] > rain_max_hour:
                            drop_list.append(i)
                        if i >= 5:
                            hour_slice = table[rain_attr][i - 5 : i + 1].to_numpy()
                            if hour_slice.all() == np.mean(hour_slice):
                                drop_list.extend(list(range(i - 5, i + 1)))
                    drop_array = np.unique(np.array(drop_list, dtype=int))
                    table = table.drop(index=drop_array)
                    drop_array_minus = table.index[table[time_attr] == -1]
                    table = table.drop(index=drop_array_minus)
                time_df_dict[stcd] = table
                # 在这里会生成csv数据表，存在副作用
                table.to_csv(cached_csv_path)
            elif (int(stcd) not in filter_list) & (os.path.exists(cached_csv_path)):
                table = pd.read_csv(cached_csv_path, engine="c")
                time_df_dict[stcd] = table
    return time_df_dict


def get_filter_data_by_space(
    time_df_dict,
    filter_list,
    station_gdf,
    csv_path,
    rain_attr="DRP",
    time_attr="TM",
    id_attr="STCD",
):
    """
    :param time_df_dict: 根据get_filter_data_by_time()得到的中间数据dict
    :param filter_list: 其他预处理过程得到的黑名单，以过滤不可用的站点
    :param station_gdf: 存储站点位置的GeoDataFrame
    :param csv_path: 按站点保存清洗后数据的文件夹
    :param rain_attr: 表格中标示降雨的属性（列名），默认为DRP
    :param time_attr: 表格中标示时间的属性（列名），默认为TM
    :param id_attr: station_gdf表格中标示站号的属性（列名），默认为STCD
    :return: 站点号与根据空间均值排除后的表格构成的dict
    """
    neighbor_stas_dict = find_neighbor_dict(station_gdf, filter_list)[0]
    space_df_dict = {}
    for key in time_df_dict:
        time_drop_list = []
        neighbor_stas = neighbor_stas_dict[key]
        table = time_df_dict[key]
        table = table.set_index(time_attr)
        for time in table.index:
            rain_time_dict = {}
            for neighbor in neighbor_stas:
                neighbor_df = time_df_dict[str(neighbor)]
                neighbor_df = neighbor_df.set_index(time_attr)
                if time in neighbor_df.index:
                    rain_time_dict[str(neighbor)] = neighbor_df[rain_attr][time]
            if not rain_time_dict:
                continue
            elif 0 < len(rain_time_dict) < 12:
                weight_rain = 0
                weight_dis = 0
                for sta in rain_time_dict:
                    point = station_gdf.geometry[
                        station_gdf[id_attr] == str(sta)
                    ].values[0]
                    point_self = station_gdf.geometry[
                        station_gdf[id_attr] == str(key)
                    ].values[0]
                    dis = distance(point, point_self)
                    weight_rain += table[rain_attr][time] / (dis**2)
                    weight_dis += 1 / (dis**2)
                interp_rain = weight_rain / weight_dis
                if abs(interp_rain - table[rain_attr][time]) > 4:
                    time_drop_list.append(time)
            elif len(rain_time_dict) >= 12:
                rain_time_series = pd.Series(rain_time_dict.values())
                quantile_25 = rain_time_series.quantile(q=0.25)
                quantile_75 = rain_time_series.quantile(q=0.75)
                average = rain_time_series.mean()
                if rain_attr in table.columns:
                    MA_Tct = (table[rain_attr][time] - average) / (
                        quantile_75 - quantile_25
                    )
                    if MA_Tct > 4:
                        time_drop_list.append(time)
        table = table.drop(index=time_drop_list).drop(columns=["Unnamed: 0"])
        space_df_dict[key] = table
        # 会生成csv文件，有副作用
        table.to_csv(os.path.join(csv_path, key + ".csv"))
    return space_df_dict


def find_neighbor_dict(station_gdf, filter_list, id_attr="STCD"):
    """
    :param station_gdf: 存储有站点位置的GeoDataFrame
    :param filter_list: 其他预处理过程得到的黑名单，以过滤不可用的站点
    :param id_attr: station_gdf表格中标示站号的属性（列名），默认为STCD
    :return: 与各站相邻的站点号（取0-0.2度）
    """
    station_gdf = station_gdf.set_index(id_attr).drop(index=filter_list).reset_index()
    station_gdf[id_attr] = station_gdf[id_attr].astype("str")
    neighbor_dict = {}
    for i in range(len(station_gdf.geometry)):
        stcd = station_gdf[id_attr][i]
        station_gdf["distance"] = station_gdf.apply(
            lambda x: distance(station_gdf.geometry[i], x.geometry), axis=1
        )
        nearest_stas = station_gdf[
            (station_gdf["distance"] > 0) & (station_gdf["distance"] <= 0.2)
        ]
        nearest_stas_list = nearest_stas[id_attr].to_list()
        neighbor_dict[stcd] = nearest_stas_list
    station_gdf = station_gdf.drop(columns=["distance"])
    return neighbor_dict
