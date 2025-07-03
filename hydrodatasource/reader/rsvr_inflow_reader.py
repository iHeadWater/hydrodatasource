"""
Author: Wenyu Ouyang
Date: 2023-10-31 09:26:31
LastEditTime: 2025-01-09 19:15:20
LastEditors: Wenyu Ouyang
Description: Reading cleaned reservoir inflow data
FilePath: \hydrodatasource\hydrodatasource\reader\rsvr_inflow_reader.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import collections
import os

import pandas as pd
from tqdm import tqdm
import xarray as xr
from hydrodatasource.reader.data_source import HydroData


class RsvrInflowReader(HydroData):
    def __init__(self, data_folder):
        self.data_source_dir = data_folder
        self.data_source_description = self.set_data_source_describe()
        self.rsvr_info = self.read_rsvr_info()

    def set_data_source_describe(self):
        return collections.OrderedDict(
            {
                "RSVR_INFO_FILE": os.path.join(self.data_source_dir, "rsvr_info.csv"),
            }
        )

    def read_rsvr_info(self):
        return pd.read_csv(
            self.data_source_description["RSVR_INFO_FILE"], dtype={"STCD": str}
        )

    def read_rsvr_inflow(self):
        # Create an empty xarray.Dataset
        rsvr_inflow_data = xr.Dataset()

        # Iterate over all reservoir IDs in rsvr_info with tqdm progress bar
        inflow_data_list = []
        for rsvr_id in tqdm(
            self.rsvr_info["STCD"], desc="Reading reservoir inflow data"
        ):
            # Read inflow data for a single reservoir
            single_rsvr_inflow = self.read_1rsvr_inflow(rsvr_id)
            inflow_data_list.append(single_rsvr_inflow)
        # check the earlyest date
        start_date = pd.Timestamp.max
        for inflow in inflow_data_list:
            if inflow.index[0] < start_date:
                start_date = inflow.index[0]
        # if the start date is too early (before 1980), we will drop the data before 1980
        if start_date.year < 1980:
            start_date = pd.Timestamp(year=1980, month=1, day=1)
        # Concatenate all inflow data along the 'STCD' dimension
        inflow_data = xr.concat(
            [
                xr.DataArray(
                    inflow.loc[start_date:]["INQ"].values,
                    dims=["time"],
                    coords={
                        "time": inflow.loc[start_date:].index.values,
                        "STCD": rsvr_id,
                    },
                )
                for inflow, rsvr_id in zip(inflow_data_list, self.rsvr_info["STCD"])
            ],
            dim="STCD",
        )

        # Add the inflow data to the xarray.Dataset
        rsvr_inflow_data["inflow"] = inflow_data

        return rsvr_inflow_data

    def read_1rsvr_inflow(self, rsvr_id):
        """
        Read inflow data for a single reservoir.

        Parameters
        ----------
        rsvr_id : str
            Reservoir ID

        Returns
        -------
        pd.DataFrame
            A DataFrame containing inflow data for the specified reservoir.
        """
        file_path = os.path.join(
            self.data_source_dir, rsvr_id, f"{rsvr_id}_rsvr_data.csv"
        )
        df = pd.read_csv(
            file_path, parse_dates=["TM"], dtype={"STCD": str, "INQ": float}
        )
        df.set_index("TM", inplace=True)

        return df
