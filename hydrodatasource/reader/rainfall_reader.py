"""
Author: liutiaxqabs 1498093445@qq.com
Date: 2025-01-15 09:39:36
LastEditors: Wenyu Ouyang
LastEditTime: 2025-01-15 14:08:05
FilePath: \hydrodatasource\hydrodatasource\reader\rainfall_reader.py
Description: Reader for rainfall data
"""

import collections
import os

import pandas as pd
from tqdm import tqdm
import xarray as xr
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


class RainfallReader(HydroData):
    def __init__(self, data_folder, output_folder):
        self.data_source_dir = data_folder
        self.output_folder = output_folder
        self.data_source_description = self.set_data_source_describe()
        self.pptn_info = self.read_pptn_info()

    def set_data_source_describe(self):
        return collections.OrderedDict(
            {
                "PPTN_INFO_FILE": os.path.join(self.data_source_dir, "pptn_info.csv"),
            }
        )

    def read_pptn_info(self):
        return pd.read_csv(
            self.data_source_description["PPTN_INFO_FILE"], dtype={"STCD": str}
        )

    def read_station_rainfall(self, abnormal=False):
        # Create an empty xarray.Dataset
        pptn_rainfall_data = xr.Dataset()

        # Iterate over all station IDs in pptn_info with tqdm progress bar
        rainfall_data_list = []
        for station_id in tqdm(
            self.pptn_info["STCD"], desc="Reading station rainfall data"
        ):
            # Read rainfall data for a single station
            single_station_rainfall = self.read_1station_rainfall(station_id)
            rainfall_data_list.append(single_station_rainfall)

        # update abnormal data
        if abnormal == True:
            abnormal_data = merge_filtered_data_recursive(
                self.output_folder, ["consistency", "extreme", "gradient"]
            )
            rainfall_data_list = filter_out_rows(
                rainfall_data_list, abnormal_data, ["STCD", "TM"]
            )

        # Check the earliest date
        start_date = pd.Timestamp.max
        for rainfall in rainfall_data_list:
            if rainfall.index[0] < start_date:
                start_date = rainfall.index[0]

        # If the start date is too early (before 1980), drop data before 1980
        if start_date.year < 1980:
            start_date = pd.Timestamp(year=1980, month=1, day=1)

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
                    rainfall_data_list, self.pptn_info["STCD"]
                )
            ],
            dim="STCD",
        )

        # Add the rainfall data to the xarray.Dataset
        pptn_rainfall_data["rainfall"] = rainfall_data

        return pptn_rainfall_data

    def read_1station_rainfall(self, station_id, abnormal=False):
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
            self.data_source_dir, station_id, f"pp_CHN_songliao_{station_id}.csv"
        )
        df = pd.read_csv(
            file_path, parse_dates=["TM"], dtype={"STCD": str, "DRP": float}
        )
        # update abnormal data
        if abnormal == True:
            abnormal_data = merge_filtered_data_recursive(
                self.output_folder, ["consistency", "extreme", "gradient"]
            )
            df = filter_out_rows(df, abnormal_data, ["STCD", "TM"])

        df.set_index("TM", inplace=True)

        return df
