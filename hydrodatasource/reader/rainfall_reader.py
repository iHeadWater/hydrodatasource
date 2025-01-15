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

class RainfallReader(HydroData):
    def __init__(self, data_folder, output_folder):
        self.data_source_dir = data_folder
        self.output_folder = output_folder
        self.data_source_description = self.set_data_source_describe()
        self.pptn_info = self.read_pptn_info()

    def set_data_source_describe(self):
        return collections.OrderedDict(
            {
                "PPTN_INFO_FILE": os.path.join(self.data_source_dir,"stations", "stations.csv"),
            }
        )

    def read_pptn_info(self):
        return pd.read_csv(
            self.data_source_description["PPTN_INFO_FILE"], dtype={"STCD": str}
        )

    def read_basin_rainfall(self, basin_id, abnormal=False):
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
        folder_path = os.path.join(self.data_source_dir, basin_id)
        all_files = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.endswith(".csv")
        ]
        return pd.concat([pd.read_csv(f) for f in all_files], ignore_index=True)
