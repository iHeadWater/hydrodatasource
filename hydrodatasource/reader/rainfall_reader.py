"""
Author: liutiaxqabs 1498093445@qq.com
Date: 2025-01-15 09:39:36
LastEditors: Wenyu Ouyang
LastEditTime: 2025-01-15 16:13:50
FilePath: \hydrodatasource\hydrodatasource\reader\rainfall_reader.py
Description: Reader for cleaned rainfall data
"""

import collections
import os

import pandas as pd
from tqdm import tqdm
import xarray as xr

from hydroutils.hydro_log import hydro_logger

from hydrodatasource.reader.data_source import HydroData


def merge_filtered_data_recursive(base_path, name_filter):
    """
    递归搜索指定路径下所有符合条件的 CSV 文件，并合并指定列（STCD, DRP, TM）。

    Parameters
    ----------
    base_path : str
        要搜索的基路径。
    name_filter : str
        文件名中需要匹配的关键字。

    Returns
    -------
    pd.DataFrame
        合并后的数据，包含列 STCD, DRP, TM。
    """
    merged_data = pd.DataFrame()

    for root, _, files in os.walk(base_path):
        for file_name in files:
            if file_name.endswith(".csv") and name_filter in file_name:
                full_path = os.path.join(root, file_name)
                # 读取文件并筛选所需列
                df = pd.read_csv(
                    full_path, usecols=["STCD", "DRP", "TM"], low_memory=False
                )
                merged_data = pd.concat([merged_data, df], ignore_index=True)

    return merged_data


def filter_out_rows(df1, df2, key_columns):
    """
    从 df1 中移除所有在 df2 中存在的行（基于 key_columns）。

    参数:
    - df1: pd.DataFrame，第一个表。
    - df2: pd.DataFrame，第二个表。
    - key_columns: list，作为对比的列名列表。

    返回:
    - pd.DataFrame: 过滤后的 df1。
    """
    # 确保两个 DataFrame 的对比列都存在
    if not all(col in df1.columns and col in df2.columns for col in key_columns):
        raise ValueError("所有指定的 key_columns 都必须存在于两个表中")

    # 使用 merge 方法找到交集
    common_rows = pd.merge(df1, df2, on=key_columns)

    # 过滤掉交集中的行
    filtered_df = df1[
        ~df1[key_columns]
        .apply(tuple, axis=1)
        .isin(common_rows[key_columns].apply(tuple, axis=1))
    ]

    return filtered_df


@hydro_logger
class RainfallReader(HydroData):
    def __init__(self, data_folder):
        """A reader for basin rainfall data
        Even for station data, we still use basin as the index

        Parameters
        ----------
        data_folder : _type_
            _description_
        """
        self.data_source_dir = data_folder
        self.data_source_description = self.set_data_source_describe()
        self.station_info = self.read_station_info()

    def set_data_source_describe(self):
        data_source_dir = self.data_source_dir
        station_rainfall_data_dir = os.path.join(
            data_source_dir, "station_rainfall_cleaned"
        )
        basin_mean_data_dir = os.path.join(data_source_dir, "basin_mean_rainfall")
        station_info_file = os.path.join(data_source_dir, "stations", "stations.csv")
        stations_csv_path = os.path.join(data_source_dir, "basins_pp_stations")
        shp_folder = os.path.join(data_source_dir, "basins_shp")
        return collections.OrderedDict(
            STATION_RAINFALL_DATA_DIR=station_rainfall_data_dir,
            BASIN_RAINFALL_DATA_DIR=basin_mean_data_dir,
            STATION_INFO_FILE=station_info_file,
            STATIONS_CSV_PATH=stations_csv_path,
            SHP_FOLDER=shp_folder,
        )

    def read_station_info(self):
        return pd.read_csv(
            self.data_source_description["STATION_INFO_FILE"], dtype={"STCD": str}
        )

    def read_basin_station_info(self, basin_id):
        basin_station_info_file = os.path.join(
            self.data_source_description["STATIONS_CSV_PATH"],
            f"{basin_id}_stations.csv",
        )
        return pd.read_csv(basin_station_info_file, dtype={"STCD": str})

    def read_basin_stations_rainfall(self, basin_id):
        # Create an empty xarray.Dataset
        pptn_rainfall_data = xr.Dataset()

        # Iterate over all station IDs in pptn_info with tqdm progress bar
        rainfall_data_list = []
        basin_station_info = self.read_basin_station_info(basin_id)
        for station_id in tqdm(
            basin_station_info["STCD"], desc="Reading station rainfall data"
        ):
            # Read rainfall data for a single station
            single_station_rainfall = self.read_basin_1station_rainfall(
                basin_id, station_id
            )
            rainfall_data_list.append(single_station_rainfall)

        start_date = self._get_start_date(rainfall_data_list)
        # Concatenate all rainfall data along the 'STCD' dimension
        rainfall_data = xr.concat(
            [
                xr.DataArray(
                    rainfall.loc[start_date:]["DRP"].values,
                    dims=["time"],
                    coords={
                        "time": rainfall.loc[start_date:].index.values,
                        "STCD": station_id,
                    },
                )
                for rainfall, station_id in zip(
                    rainfall_data_list, basin_station_info["STCD"]
                )
            ],
            dim="STCD",
        )

        pptn_rainfall_data[basin_id] = rainfall_data

        return pptn_rainfall_data

    def read_basin_1station_rainfall(self, basin_id, station_id):
        """
        Read rainfall data for a single station.

        Parameters
        ----------
        station_id : str
            Station ID

        Returns
        -------
        pd.DataFrame
            A DataFrame containing rainfall data for the specified station.
        """
        file_path = os.path.join(
            self.data_source_description["STATION_RAINFALL_DATA_DIR"],
            basin_id,
            # TODO: songliao is a hard code -- prefix may change in different data source
            f"pp_CHN_songliao_{station_id}.csv",
        )
        df = pd.read_csv(
            file_path, parse_dates=["TM"], dtype={"STCD": str, "DRP": float}
        )
        df.set_index("TM", inplace=True)

        return df

    def read_1basin_rainfall(self, basin_id):
        """
        Read basin-mean rainfall data for a single basin.

        Parameters
        ----------
        basin_id : str
            Basin ID

        Returns
        -------
        pd.DataFrame
            A DataFrame containing rainfall data for the specified basin.
        """
        basin_mean_file = os.path.join(
            self.data_source_description["BASIN_RAINFALL_DATA_DIR"],
            f"{basin_id}_rainfall.csv",
        )
        basin_rainfall_data = pd.read_csv(
            basin_mean_file, parse_dates=["TM"], dtype={"mean_rainfall": float}
        )
        # rename col mean_rainfall to rain
        basin_rainfall_data.rename(columns={"mean_rainfall": "rain"}, inplace=True)
        # set TM as index
        basin_rainfall_data.set_index("TM", inplace=True)
        return basin_rainfall_data

    def read_basin_rainfall(self, basin_ids):
        """
        Read rainfall data for all basins.

        Parameters
        ----------
        basin_ids : list
            List of basin IDs

        Returns
        -------
        pd.DataFrame
            A DataFrame containing rainfall data for all basins.
        """
        rainfall_data_list = []
        for basin_id in basin_ids:
            a_basin_data = self.read_1basin_rainfall(basin_id)
            rainfall_data_list.append(a_basin_data)
        start_date = self._get_start_date(rainfall_data_list)
        return xr.concat(
            [
                xr.DataArray(
                    rainfall.loc[start_date:]["rain"].values,
                    dims=["time"],
                    coords={
                        "time": rainfall.loc[start_date:].index.values,
                        "basin": station_id,
                    },
                )
                for rainfall, station_id in zip(rainfall_data_list, basin_ids)
            ],
            dim="basin",
        )

    def _get_start_date(self, rainfall_data_list):
        result = pd.Timestamp.max
        for rainfall in rainfall_data_list:
            if rainfall.index[0] < result:
                result = rainfall.index[0]
        if result.year < 1980:
            result = pd.Timestamp(year=1980, month=1, day=1)
        return result

    def read_basin_rainfall_statistics(self):
        pass
