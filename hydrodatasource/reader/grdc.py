"""
Author: Wenyu Ouyang
Date: 2023-11-03 09:16:41
LastEditTime: 2025-01-02 20:40:12
LastEditors: Wenyu Ouyang
Description: Reading GRDC data
FilePath: \hydrodatasource\hydrodatasource\reader\grdc.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import collections
import os
import datetime
import logging
from pathlib import Path
import re
import pandas as pd
import xarray as xr
import geopandas as gpd
from dateutil.parser import parse

from hydrodatasource.configs.config import CACHE_DIR, SETTING
from hydrodatasource.reader.data_source import HydroData


class Grdc(HydroData):
    """Reading GRDC streamflow data."""

    def __init__(self, data_path):
        self.data_source_dir = data_path
        self.data_source_description = self.set_data_source_describe()
        self.grdc_site_info = self.read_site_info()

    def get_name(self):
        return "GRDC"

    def set_data_source_describe(self):
        data_root_dir = self.data_source_dir
        continents = [
            "africa",
            "asia",
            "south_america",
            "north_america",
            "south_west_pacific",
            "europe",
        ]
        shape_dir = os.path.join(data_root_dir, "GRDC_Watersheds")

        data_description = {}
        for continent in continents:
            continent_dir = os.path.join(data_root_dir, continent)
            daily_dir = os.path.join(continent_dir, "daily")
            monthly_dir = os.path.join(continent_dir, "monthly")

            data_description[continent] = {
                "daily": daily_dir,
                "monthly": monthly_dir,
            }

        return collections.OrderedDict(
            DATA_DIR=data_root_dir,
            CONTINENT_DATA=data_description,
            SHAPE_DIR=shape_dir,
        )

    def read_site_info(self):
        """Reads the shapefile and extracts the 'id' column as a list."""
        shape_dir = self.data_source_description["SHAPE_DIR"]
        shp_files = [f for f in os.listdir(shape_dir) if f.endswith(".shp")]

        if len(shp_files) != 1:
            raise ValueError(
                f"Expected one shapefile in {shape_dir}, found {len(shp_files)}"
            )

        shp_file_path = os.path.join(shape_dir, shp_files[0])
        gdf = gpd.read_file(shp_file_path)
        if "grdc_no" not in gdf.columns:
            raise ValueError(
                f"The shapefile {shp_file_path} does not contain an 'grdc_no' column."
            )
        gdf["grdc_no"] = gdf["grdc_no"].apply(
            lambda x: str(int(x)) if isinstance(x, float) else str(x)
        )
        assert all(x < y for x, y in zip(gdf["grdc_no"], gdf["grdc_no"][1:]))
        return gdf[["grdc_no", "area"]]

    def map_station_to_continent(self, station_id: str):
        """Maps a station ID to its corresponding continent based on rules."""
        if station_id.startswith("1"):
            return "africa"
        elif station_id.startswith("2"):
            return "asia"
        elif station_id.startswith("3"):
            return "south_america"
        elif station_id.startswith("4"):
            return "north_america"
        elif station_id.startswith("5"):
            return "south_west_pacific"
        elif station_id.startswith("6"):
            return "europe"
        else:
            raise ValueError(f"Invalid station ID: {station_id}")

    def read_grdc_daily_data(
        self,
        station_id: str,
        time_range: list,
        parameter: str = "Q",
        column: str = "streamflow",
    ):
        """read daily river discharge data from Global Runoff Data Centre (GRDC).

        Requires the GRDC daily data files in a local directory. The GRDC daily data
        files can be ordered at
        https://www.bafg.de/GRDC/EN/02_srvcs/21_tmsrs/riverdischarge_node.html

        Parameters
        ----------
            station_id: The station id to get. The station id can be found in the
                catalogues at
                https://www.bafg.de/GRDC/EN/02_srvcs/21_tmsrs/212_prjctlgs/project_catalogue_node.html
            time_range: A list of [start_time, end_time] in UTC and ISO format strings e.g.
                ['YYYY-MM-DDTHH:MM:SSZ', 'YYYY-MM-DDTHH:MM:SSZ'].
            parameter: optional. The parameter code to get, e.g. ('Q') discharge,
                cubic meters per second.
            column: optional. Name of column in dataframe. Default: "streamflow".

        Returns:
            grdc data in a dataframe and metadata.

        Examples:
            .. code-block:: python

                from ewatercycle.observation.grdc import get_grdc_data

                df, meta = get_grdc_data('6335020',
                                        ['2000-01-01T00:00Z', '2001-01-01T00:00Z'])
                df.describe()
                        streamflow
                count   4382.000000
                mean    2328.992469
                std     1190.181058
                min      881.000000
                25%     1550.000000
                50%     2000.000000
                75%     2730.000000
                max    11300.000000

                meta
                {'grdc_file_name': '/home/myusername/git/eWaterCycle/ewatercycle/6335020_Q_Day.Cmd.txt',
                'id_from_grdc': 6335020,
                'file_generation_date': '2019-03-27',
                'river_name': 'RHINE RIVER',
                'station_name': 'REES',
                'country_code': 'DE',
                'grdc_latitude_in_arc_degree': 51.756918,
                'grdc_longitude_in_arc_degree': 6.395395,
                'grdc_catchment_area_in_km2': 159300.0,
                'altitude_masl': 8.0,
                'dataSetContent': 'MEAN DAILY DISCHARGE (Q)',
                'units': 'm³/s',
                'time_series': '1814-11 - 2016-12',
                'no_of_years': 203,
                'last_update': '2018-05-24',
                'nrMeasurements': 'NA',
                'UserStartTime': '2000-01-01T00:00Z',
                'UserEndTime': '2001-01-01T00:00Z',
                'nrMissingData': 0}
        """  # noqa: E501
        continent = self.map_station_to_continent(station_id)
        data_path = self.data_source_description["CONTINENT_DATA"][continent]["daily"]

        if not os.path.exists(data_path):
            raise ValueError(f"The GRDC daily directory {data_path} does not exist!")

        # Read the raw data
        start_time, end_time = time_range
        raw_file = os.path.join(data_path, f"{station_id}_{parameter}_Day.Cmd.txt")
        if not os.path.exists(raw_file):
            raise ValueError(f"The GRDC file {raw_file} does not exist!")

        # Convert the raw data to an xarray
        metadata, df = _grdc_read(
            raw_file,
            start=_get_time(start_time).date(),
            end=_get_time(end_time).date(),
            column=column,
        )

        # Add start/end_time to metadata
        metadata["UserStartTime"] = start_time
        metadata["UserEndTime"] = end_time

        # Add number of missing data to metadata
        metadata["nrMissingData"] = _count_missing_data(df, column)

        # Show info about data
        _log_metadata(metadata)

        return df, metadata

    def cache_grdc_daily(self, station_ids=None, time_range=None, batch_size=1000):
        """
        Save GRDC daily data to a NetCDF file.

        Parameters
        ----------
        station_ids: list of str
            List of station IDs to read data for.
        time_range: list of str
            List of [start_time, end_time] in UTC and ISO format strings e.g.
            ['YYYY-MM-DDTHH:MM:SSZ', 'YYYY-MM-DDTHH:MM:SSZ'].
        batch_size: int
            Number of stations to process in each batch
        """
        if time_range is None:
            time_range = ["1980-01-01", "2023-12-31"]
        start_date, end_date = time_range
        catalogue = self.grdc_site_info
        if station_ids is None:
            station_ids = catalogue["grdc_no"].tolist()
        catalogue = catalogue[catalogue["grdc_no"].isin(station_ids)]

        def data_generator(station_ids, batch_size):
            for i in range(0, len(station_ids), batch_size):
                yield station_ids[i : i + batch_size]

        # Split station IDs into batches
        station_batches = list(data_generator(station_ids, batch_size))

        # Loop over each station in the catalogue
        for batch_index, batch in enumerate(station_batches):
            # Create empty lists to store data and metadata
            data_list = []
            meta_list = []
            for station_id in batch:
                try:
                    st = datetime.datetime.strptime(start_date, "%Y-%m-%d").strftime(
                        "%Y-%m-%dT%H:%M:%SZ"
                    )
                    et = datetime.datetime.strptime(end_date, "%Y-%m-%d").strftime(
                        "%Y-%m-%dT%H:%M:%SZ"
                    )
                    df, meta = self.read_grdc_daily_data(str(station_id), [st, et])
                except Exception as e:
                    print(f"Error reading data for station {station_id}: {e}")
                    # Create an empty DataFrame with the same structure
                    df = pd.DataFrame(
                        columns=["streamflow"],
                        index=pd.date_range(start=start_date, end=end_date),
                    )
                    df["streamflow"] = float("nan")
                    meta = {
                        "grdc_file_name": "",
                        "id_from_grdc": station_id,
                        "river_name": "",
                        "station_name": "",
                        "country_code": "",
                        "grdc_latitude_in_arc_degree": float("nan"),
                        "grdc_longitude_in_arc_degree": float("nan"),
                        "grdc_catchment_area_in_km2": float("nan"),
                        "altitude_masl": float("nan"),
                        "dataSetContent": "",
                        "units": "m³/s",
                        "time_series": "",
                        "no_of_years": 0,
                        "last_update": "",
                        "nrMeasurements": "NA",
                        "UserStartTime": start_date,
                        "UserEndTime": end_date,
                        "nrMissingData": _count_missing_data(df, "streamflow"),
                    }

                coords_dict = {"time": df.index, "station": [station_id]}
                da = xr.DataArray(
                    data=df["streamflow"].values.reshape(-1, 1),
                    coords=coords_dict,
                    dims=["time", "station"],
                    name="streamflow",
                    attrs={"units": meta.get("units", "unknown")},
                )
                data_list.append(da)

                # Append metadata
                meta_list.append(meta)

            # Concatenate all DataArrays along the 'station' dimension
            ds = xr.concat(data_list, dim="station")

            # Assign attributes
            ds.attrs["description"] = "Daily river discharge"
            ds.station.attrs["description"] = "GRDC station number"

            # Write the xarray Dataset to a NetCDF file
            batch_file_path = os.path.join(
                CACHE_DIR,
                f"grdc_daily_streamflow_data_batch_{batch[0]}_{batch[-1]}.nc",
            )
            ds.to_netcdf(batch_file_path)
            print(
                f"Batch {batch_index + 1}/{len(station_batches)} NetCDF file created successfully!"
            )
            # Release memory by deleting the dataset
            del ds
            del data_list
            del meta_list

    def read_streamflow_xrdataset(
        self,
        station_id_lst: list = None,
        time_range: list = None,
        **kwargs,
    ) -> dict:
        """
        Read GRDC daily data from multiple NetCDF files and organize them by station IDs.

        Parameters
        ----------
        station_id_lst : list
            List of station IDs to select.
        time_range : list
            List of two elements [start_time, end_time] to select time range.
        **kwargs: Additional arguments.

        Returns
        ----------
        dict: A dictionary where each key is a station ID and each value is an xarray.Dataset containing the selected station IDs, time range, and variable.
        """
        if station_id_lst is None or time_range is None:
            return None

        # Initialize a dictionary to hold datasets for each station ID
        datasets_by_station_id = {}

        # Collect batch files
        batch_files = [
            os.path.join(CACHE_DIR, f)
            for f in os.listdir(CACHE_DIR)
            if re.match(
                rf"^grdc_daily_streamflow_data_batch_[A-Za-z0-9_]+_[A-Za-z0-9_]+\.nc$",
                f,
            )
        ]

        if not batch_files:
            # Cache the data if no batch files are found
            self.cache_grdc_daily(
                station_ids=station_id_lst, time_range=time_range, **kwargs
            )
            batch_files = [
                os.path.join(CACHE_DIR, f)
                for f in os.listdir(CACHE_DIR)
                if re.match(
                    rf"^grdc_daily_streamflow_data_batch_[A-Za-z0-9_]+_[A-Za-z0-9_]+\.nc$",
                    f,
                )
            ]

        selected_datasets = []

        for batch_file in batch_files:
            ds = xr.open_dataset(batch_file)
            if valid_station_ids := [
                sid for sid in station_id_lst if sid in ds["station"].values
            ]:
                ds_selected = ds[["streamflow"]].sel(
                    station=valid_station_ids, time=slice(time_range[0], time_range[1])
                )
                selected_datasets.append(ds_selected)

            ds.close()  # Close the dataset to free memory

        # If any datasets were selected, concatenate them along the 'station' dimension
        if selected_datasets:
            datasets_by_station_id = xr.concat(selected_datasets, dim="station").sortby(
                "station"
            )
        else:
            datasets_by_station_id = xr.Dataset()

        return datasets_by_station_id


def _get_time(time_iso: str) -> datetime.datetime:
    """Return a datetime in UTC.
    Convert a date string in ISO format to a datetime
    and check if it is in UTC.
    """
    time = parse(time_iso)
    if time.tzname() != "UTC":
        raise ValueError(
            "The time is not in UTC. The ISO format for a UTC time "
            "is 'YYYY-MM-DDTHH:MM:SSZ'"
        )
    return time


def _grdc_metadata_reader(grdc_station_path, all_lines):
    """
    Initiating a dictionary that will contain all GRDC attributes.
    This function is based on earlier work by Rolf Hut.
    https://github.com/RolfHut/GRDC2NetCDF/blob/master/GRDC2NetCDF.py
    DOI: 10.5281/zenodo.19695
    that function was based on earlier work by Edwin Sutanudjaja from Utrecht University.
    https://github.com/edwinkost/discharge_analysis_IWMI
    Modified by Susan Branchett
    """

    # split the content of the file into several lines
    all_lines = all_lines.replace("\r", "")
    all_lines = all_lines.split("\n")

    # get grdc ids (from files) and check their consistency with their
    # file names
    id_from_file_name = int(
        os.path.basename(grdc_station_path).split(".")[0].split("_")[0]
    )
    id_from_grdc = None
    if id_from_file_name == int(all_lines[8].split(":")[1].strip()):
        id_from_grdc = int(all_lines[8].split(":")[1].strip())
    else:
        print(
            f"GRDC station {id_from_file_name} ({str(grdc_station_path)}) is NOT used."
        )

    attribute_grdc = {}
    if id_from_grdc is not None:
        attribute_grdc["grdc_file_name"] = str(grdc_station_path)
        attribute_grdc["id_from_grdc"] = id_from_grdc

        try:
            attribute_grdc["file_generation_date"] = str(
                all_lines[6].split(":")[1].strip()
            )
        except (IndexError, ValueError):
            attribute_grdc["file_generation_date"] = "NA"

        try:
            attribute_grdc["river_name"] = str(all_lines[9].split(":")[1].strip())
        except (IndexError, ValueError):
            attribute_grdc["river_name"] = "NA"

        try:
            attribute_grdc["station_name"] = str(all_lines[10].split(":")[1].strip())
        except (IndexError, ValueError):
            attribute_grdc["station_name"] = "NA"

        try:
            attribute_grdc["country_code"] = str(all_lines[11].split(":")[1].strip())
        except (IndexError, ValueError):
            attribute_grdc["country_code"] = "NA"

        try:
            attribute_grdc["grdc_latitude_in_arc_degree"] = float(
                all_lines[12].split(":")[1].strip()
            )
        except (IndexError, ValueError):
            attribute_grdc["grdc_latitude_in_arc_degree"] = "NA"

        try:
            attribute_grdc["grdc_longitude_in_arc_degree"] = float(
                all_lines[13].split(":")[1].strip()
            )
        except (IndexError, ValueError):
            attribute_grdc["grdc_longitude_in_arc_degree"] = "NA"

        try:
            attribute_grdc["grdc_catchment_area_in_km2"] = float(
                all_lines[14].split(":")[1].strip()
            )
            if attribute_grdc["grdc_catchment_area_in_km2"] <= 0.0:
                attribute_grdc["grdc_catchment_area_in_km2"] = "NA"
        except (IndexError, ValueError):
            attribute_grdc["grdc_catchment_area_in_km2"] = "NA"

        try:
            attribute_grdc["altitude_masl"] = float(all_lines[15].split(":")[1].strip())
        except (IndexError, ValueError):
            attribute_grdc["altitude_masl"] = "NA"

        try:
            attribute_grdc["dataSetContent"] = str(all_lines[21].split(":")[1].strip())
        except (IndexError, ValueError):
            attribute_grdc["dataSetContent"] = "NA"

        try:
            attribute_grdc["units"] = str(all_lines[23].split(":")[1].strip())
        except (IndexError, ValueError):
            attribute_grdc["units"] = "NA"

        try:
            attribute_grdc["time_series"] = str(all_lines[24].split(":")[1].strip())
        except (IndexError, ValueError):
            attribute_grdc["time_series"] = "NA"

        try:
            attribute_grdc["no_of_years"] = int(all_lines[25].split(":")[1].strip())
        except (IndexError, ValueError):
            attribute_grdc["no_of_years"] = "NA"

        try:
            attribute_grdc["last_update"] = str(all_lines[26].split(":")[1].strip())
        except (IndexError, ValueError):
            attribute_grdc["last_update"] = "NA"

        try:
            attribute_grdc["nrMeasurements"] = int(
                str(all_lines[34].split(":")[1].strip())
            )
        except (IndexError, ValueError):
            attribute_grdc["nrMeasurements"] = "NA"

    return attribute_grdc


def _grdc_read(grdc_station_file, start, end, column):
    grdc_station_path = Path(grdc_station_file)
    with grdc_station_path.open("r", encoding="cp1252", errors="ignore") as file:
        data = file.read()

    metadata = _grdc_metadata_reader(grdc_station_path, data)

    all_lines = data.split("\n")
    header = next(
        (i + 1 for i, line in enumerate(all_lines) if line.startswith("# DATA")),
        0,
    )
    # Import GRDC data into dataframe and modify dataframe format
    grdc_data = pd.read_csv(
        grdc_station_path,
        encoding="cp1252",
        skiprows=header,
        delimiter=";",
        parse_dates=["YYYY-MM-DD"],
        na_values="-999",
    )
    grdc_station_df = pd.DataFrame(
        {column: grdc_data[" Value"].array},
        index=grdc_data["YYYY-MM-DD"].array,
    )
    grdc_station_df.index.rename("time", inplace=True)

    # Create a continuous date range based on the given start and end dates
    full_date_range = pd.date_range(start=start, end=end)
    full_df = pd.DataFrame(index=full_date_range)
    full_df.index.rename("time", inplace=True)

    # Merge the two dataframes, so the dates without data will have NaN values
    merged_df = full_df.merge(
        grdc_station_df, left_index=True, right_index=True, how="left"
    )

    return metadata, merged_df


def _count_missing_data(df, column):
    """Return number of missing data."""
    return df[column].isna().sum()


def _log_metadata(metadata):
    """Print some information about data."""
    logger = logging.getLogger(__name__)
    coords = (
        metadata["grdc_latitude_in_arc_degree"],
        metadata["grdc_longitude_in_arc_degree"],
    )
    message = (
        f"GRDC station {metadata['id_from_grdc']} is selected. "
        f"The river name is: {metadata['river_name']}."
        f"The coordinates are: {coords}."
        f"The catchment area in km2 is: {metadata['grdc_catchment_area_in_km2']}. "
        f"There are {metadata['nrMissingData']} missing values during "
        f"{metadata['UserStartTime']}_{metadata['UserEndTime']} at this station. "
        f"See the metadata for more information."
    )
    logger.info("%s", message)


if __name__ == "__main__":
    data_dir = os.path.join(SETTING["local_data_path"]["datasets-origin"], "GRDC")
    grdc = Grdc(data_dir)
    # grdc.cache_grdc_daily(["1107700", "4101200"], ["1990-10-01", "2000-10-01"])
    # grdc.cache_grdc_daily()
    grdc.read_streamflow_xrdataset(["1107700", "4101200"], ["1990-10-01", "2000-10-01"])
