"""
Author: Wenyu Ouyang
Date: 2024-03-24 08:48:57
LastEditTime: 2024-03-28 08:38:13
LastEditors: Wenyu Ouyang
Description: Check user's data format is correct (local and minio)
FilePath: \hydrodatasource\hydrodatasource\processor\data_checker.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import os
import pandas as pd
from hydrodatasource.configs.config import (
    LOCAL_DATA_PATH,
)


class DataChecker:
    """
    A class to check the format of station, reservoir, and basin data based on specified rules.
    """

    def __init__(self):
        """
        Initialize the DataChecker with the path to the data.
        """
        self.base_path = LOCAL_DATA_PATH
        # TODO: all specifications should be checked again
        # NOTE: we prioritize the basin data format check as it is directly related to the model
        self.expected_structure = {
            "stations-origin": [
                "pp_stations",
                "zq_stations",
                "zz_stations",
                "stations_list",
            ],
            "reservoirs-origin": ["day_data", "hour_data", "reservoirs_list"],
            "basins-origin": [
                "basins_list",
                "basin_shapefiles",
                "hour_data",
                "attributes.nc",
                "basins_shp.zip",
                "HydroRIVERS_v10_shp.zip",
            ],
        }
        self.expected_columns = {
            # 站点基础信息表
            "stations_list": [
                "ID",
                "STCD",
                "STTYPE",
                "VARTYPE",
                "DIVISION",
                "LON",
                "LAT",
            ],
            # 雨量站时序数据表
            "pp_stations": ["ID", "STCD", "TM", "DRP", "INTV", "PDR", "DYP", "WTH"],
            # 河道水文站/水位站时序信息表
            "zq_stations": [
                "ID",
                "STCD",
                "TM",
                "Z",
                "Q",
                "XSA",
                "XSAVV",
                "XSMXV",
                "FLWCHRCD",
                "WPTN",
                "MSQMT",
                "MSAMT",
                "MSVMT",
            ],
            "zz_stations": [
                "ID",
                "STCD",
                "TM",
                "Z",
                "Q",
                "XSA",
                "XSAVV",
                "XSMXV",
                "FLWCHRCD",
                "WPTN",
                "MSQMT",
                "MSAMT",
                "MSVMT",
            ],
            # 水库基础信息表
            "reservoirs_list": [
                "ID",
                "STCD",
                "TTCP",
                "DDZ",
                "NORMZ",
                "FSLTDZ",
                "LON",
                "LAT",
                "BASIN",
            ],
            # 水库时序数据表
            "reservoirs": [
                "ID",
                "STCD",
                "TM",
                "RZ",
                "INQ",
                "W",
                "BLRZ",
                "OTQ",
                "RWCHRCD",
                "RWPTN",
                "INQDR",
                "MSQMT",
            ],
            # 流域属性数据
            "basins_attributes": [
                "basin",
                "area",
                "ele_mt_smn",
                "slp_dg_sav",
                "sgr_dk_sav",
                "for_pc_sse",
                "glc_cl_smj",
                "run_mm_syr",
                "inu_pc_slt",
                "cmi_ix_syr",
                "aet_mm_syr",
                "snw_pc_syr",
                "swc_pc_syr",
                "gwt_cm_sav",
                "cly_pc_sav",
                "dor_pc_pva",
            ],
            # 流域时序平均数据表
            "basins_mean_data": [
                # 根据文档提供的信息定义列名，例如:
                "BAS_CODE",
                "TS",
                "GPM_TP",
                "GFS_TP",
                "GFS_10U",
                "GFS_10V",
                "GFS_2SH",
                "GFS_2R",
                "GFS_DSWRF",
                "GFS_PWAT",
                "GFS_2T",
                "GFS_TCC",
                "SMP",
                "Q",
                # 注意：这里的列名需要根据文档中的实际定义进行调整
            ],
            # 流域时序网格数据
            "basins_grid_data": [
                # 根据文档中的具体格式定义列名
                # 例如:
                "BAS_CODE",
                "TS",
                "GRID_DATA_TYPE",
                "GRID_VALUE",
                # 注意：这里的列名需要根据文档中的实际定义进行调整
            ],
            # others should be check again and more need to be added
        }

    def check_folder_structure(self):
        """
        Checks if the folder structure in the base directory matches the expected structure.

        Returns
        -------
        bool
            True if the structure matches, False otherwise.
        """
        for main_folder, subfolders in self.expected_structure.items():
            main_folder_path = os.path.join(self.base_path, main_folder)
            if not os.path.exists(main_folder_path):
                print(f"Missing main folder: {main_folder}")
                return False
            for subfolder in subfolders:
                subfolder_path = os.path.join(main_folder_path, subfolder)
                if not os.path.exists(subfolder_path):
                    print(f"Missing subfolder: {subfolder} in {main_folder}")
                    return False
        return True

    def check_station_data_files(self, station_types=None):
        """
        Checks the format of station data files including both basic info and time-series data.
        """
        if station_types is None:
            # one can specify the station types to check,
            # such as ["pp"], ["zz"], or ["zq"]
            station_types = ["pp", "zz", "zq"]
        is_correct = True
        for station_type in station_types:
            # Check basic info file
            basic_info_file = os.path.join(
                # here we use / rather than os.path.join to avoid path issues for minio
                self.base_path,
                f"stations-origin/{station_type}_stations.csv",
            )
            if not self.check_file_format(
                basic_info_file, self.expected_columns[f"{station_type}_stations"]
            ):
                print(f"Basic info file format error: {basic_info_file}")
                is_correct = False

            # Check time-series data files
            time_series_folder = os.path.join(
                self.base_path, f"stations-origin/{station_type}_stations"
            )
            if not self.check_files_in_folder(
                time_series_folder, self.expected_columns[f"{station_type}_stations"]
            ):
                print(f"Time series file format error in folder: {time_series_folder}")
                is_correct = False

        return is_correct

    def check_file_format(self, file_path, expected_columns):
        """
        Checks the format of a given file.

        Parameters
        ----------
        file_path : str
            Path to the file.
        expected_columns : list
            List of expected column names.

        Returns
        -------
        bool
            True if the file format is correct, False otherwise.
        """
        try:
            data = pd.read_csv(file_path)
            return all(column in data.columns for column in expected_columns)
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return False

    def check_files_in_folder(self, folder_path, expected_columns):
        """
        Checks the format of all CSV files in a folder.

        Parameters
        ----------
        folder_path : str
            Path to the folder containing the files.
        expected_columns : list
            List of expected column names for the files.

        Returns
        -------
        bool
            True if all file formats are correct, False otherwise.
        """
        is_correct = True
        for file in os.listdir(folder_path):
            if file.endswith(".csv"):
                file_path = os.path.join(folder_path, file)
                if not self.check_file_format(file_path, expected_columns):
                    print(f"File format error: {file_path}")
                    is_correct = False
        return is_correct

    def check_basin_data_files(self):
        """
        Checks the format of basin data files including attributes and time-series data.
        """
        is_correct = True

        # Check basin attributes data file
        basin_attributes_file = os.path.join(
            self.base_path, "basins-origin/attributes.nc"
        )
        if not os.path.isfile(basin_attributes_file):
            print(f"Missing basin attributes data file: {basin_attributes_file}")
            is_correct = False

        # Check basin time-series data file
        basin_time_series_file = os.path.join(
            self.base_path, "basins-origin/basins_mean_data.csv"
        )
        if not self.check_file_format(
            basin_time_series_file, self.expected_columns["basins_mean_data"]
        ):
            print(f"Basin time series data file format error: {basin_time_series_file}")
            is_correct = False

        # Check for spatial data files, if required
        # Example: checking for a specific shapefile
        basin_shapefile = os.path.join(self.base_path, "basins-origin/basins_shp.zip")
        if not os.path.isfile(basin_shapefile):
            print(f"Missing basin shapefile: {basin_shapefile}")
            is_correct = False

        return is_correct

    def check_basin_average_time_series_data(self):
        """
        Checks the format of basin average time-series data file.
        """
        # Define the file path for basin average time-series data
        basin_avg_time_series_file = os.path.join(
            self.base_path, "basins-origin/basin_average_time_series.csv"
        )

        # Check if the file exists
        if not os.path.isfile(basin_avg_time_series_file):
            print(
                f"Missing basin average time series data file: {basin_avg_time_series_file}"
            )
            return False

        # Check the file format
        if not self.check_file_format(
            basin_avg_time_series_file, self.expected_columns["basins_mean_data"]
        ):
            print(
                f"Basin average time series data file format error: {basin_avg_time_series_file}"
            )
            return False

        return True
