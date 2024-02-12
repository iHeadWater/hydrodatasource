"""
Author: Wenyu Ouyang
Date: 2023-11-03 09:16:41
LastEditTime: 2023-11-08 14:12:36
LastEditors: Wenyu Ouyang
Description: Preprocess scripts for hydrostations data
FilePath: \hydro_opendata\hydro_opendata\processor\preprocess_grdc.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""
import json
import os
import re
import argparse
import datetime
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
from uu import Error
import zipfile
from dateutil.parser import parse
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point
import requests
from hydrodataset import CACHE_DIR


# Path to the cache file
cache_file_path = CACHE_DIR.joinpath("country_phone_codes.json")
STATION_LST_COLUMNS = [
    "ID",
    "STCD",
    "STNAME",
    "STTYPE",
    "VARTYPE",
    "AGENCY",
    "DIVISION",
    "LON",
    "LAT",
    "SOURCE",
]
BASIN_LST_COLUMNS = [
    "ID",
    "NAME",
    "LON",
    "LAT",
    "AREA",
    "SOURCE",
]


def fetch_country_phone_codes():
    # If the cache file exists, load the data from it
    if os.path.exists(cache_file_path):
        with open(cache_file_path, "r") as file:
            country_phone_codes = json.load(file)
        return country_phone_codes

    # If the cache file doesn't exist, fetch the data from the API
    url = "https://restcountries.com/v3.1/all"
    response = requests.get(url)

    if response.status_code != 200:
        raise Error(f"Error fetching data: {response.status_code}")
    countries = response.json()
    country_phone_codes = {
        country["name"]["common"]: re.sub(
            r"\D", "", country["idd"]["root"] + country["idd"]["suffixes"][0]
        )
        if "idd" in country
        and "root" in country["idd"]
        and "suffixes" in country["idd"]
        and country["idd"]["suffixes"]
        else "Unknown"
        for country in countries
    }

    # Save the data to the cache file
    with open(cache_file_path, "w") as file:
        json.dump(country_phone_codes, file)

    return country_phone_codes


# Use the function
country_phone_codes = fetch_country_phone_codes()
world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))


def save_and_zip_grdc_shp(gdf, basin_id, output_dir):
    # Load the countries dataset from geopandas

    # Sample point (longitude, latitude)
    longitude, latitude = gdf["long_org"].iloc[0], gdf["lat_org"].iloc[0]

    # Create a Point object from the coordinates
    point = Point(longitude, latitude)

    # Find the country that contains the point
    country = world[world.geometry.contains(point)]

    phone_code = country_phone_codes.get(country.iloc[0]["name"], "Unknown")
    station_name = gdf["station"].iloc[0]
    file_name = f"{phone_code}_grdc{basin_id}_{station_name}"
    gdf["grdc_no"] = [
        pc + "_grdc" + sc for pc, sc in zip([phone_code], gdf["grdc_no"].values)
    ]
    # change columns to uniform names
    # pp may means post-processed, while org means original, hence we use pp
    column_mapping = dict(
        zip(
            ["grdc_no", "station", "long_pp", "lat_pp", "area", "source"],
            BASIN_LST_COLUMNS,
        )
    )
    gdf = gdf.rename(columns=column_mapping)
    other_columns = [col for col in gdf.columns if col not in BASIN_LST_COLUMNS]
    new_order = BASIN_LST_COLUMNS + other_columns
    gdf = gdf[new_order]
    return save_shp(gdf, output_dir, file_name)


def save_shp(gdf, output_dir, file_name):
    shapefile_path = os.path.join(output_dir, file_name)
    # Save the GeoDataFrame as a Shapefile
    gdf.to_file(shapefile_path)

    # Zip the Shapefile components directly
    zip_path = os.path.join(output_dir, f"{file_name}.zip")
    with zipfile.ZipFile(zip_path, "w") as zipf:
        for extension in [".shp", ".shx", ".dbf", ".prj", ".cpg"]:
            file_path = os.path.join(
                f"{shapefile_path}",
                f"{file_name}{extension}",
            )
            if os.path.exists(file_path):
                zipf.write(file_path, os.path.basename(file_path))
        # Remove the temporary Shapefile components
    for extension in [".shp", ".shx", ".dbf", ".prj", ".cpg"]:
        file_path = os.path.join(f"{shapefile_path}", f"{file_name}{extension}")
        if os.path.exists(file_path):
            os.remove(file_path)
    os.removedirs(shapefile_path)
    return zip_path


# Create a directory for the basin shapefiles and their ZIPs
def onegeojson2basinsshp4grdc(gjson_path, final_zip_dir):
    # Create a directory for the final ZIP files
    os.makedirs(final_zip_dir, exist_ok=True)
    # Load the GeoJSON file
    basins_gdf = gpd.read_file(gjson_path)
    basins_gdf["grdc_no"] = basins_gdf["grdc_no"].astype(int).astype(str).str.zfill(8)

    # Iterate through each basin and create individual Shapefiles and ZIPs
    for index, row in basins_gdf.iterrows():
        basin_id = row["grdc_no"]
        basin_gdf = basins_gdf[basins_gdf.index == index]
        save_and_zip_grdc_shp(basin_gdf, basin_id, final_zip_dir)


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


def _to_absolute_path(
    input_path: str,
    parent: Optional[Path] = None,
    must_exist: bool = False,
    must_be_in_parent=True,
) -> Path:
    """Parse input string as :py:class:`pathlib.Path` object.
    Args:
        input_path: Input string path that can be a relative or absolute path.
        parent: Optional parent path of the input path
        must_exist: Optional argument to check if the input path exists.
        must_be_in_parent: Optional argument to check if the input path is
            subpath of parent path
    Returns:
        The input path that is an absolute path and a :py:class:`pathlib.Path` object.
    """
    pathlike = Path(input_path)
    if parent:
        pathlike = parent.joinpath(pathlike)
        if must_be_in_parent:
            try:
                pathlike.relative_to(parent)
            except ValueError as e:
                raise ValueError(
                    f"Input path {input_path} is not a subpath of parent {parent}"
                ) from e

    return pathlike.expanduser().resolve(strict=must_exist)


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


def _grdc_read(grdc_station_path, start, end, column):
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


logger = logging.getLogger(__name__)


def _log_metadata(metadata):
    """Print some information about data."""
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


MetaDataType = Dict[str, Union[str, int, float]]


def read_grdc_daily_data(
    station_id: str,
    start_time: str,
    end_time: str,
    data_home: Optional[str],
    parameter: str = "Q",
    column: str = "streamflow",
) -> Tuple[pd.core.frame.DataFrame, MetaDataType]:
    """read daily river discharge data from Global Runoff Data Centre (GRDC).

    Requires the GRDC daily data files in a local directory. The GRDC daily data
    files can be ordered at
    https://www.bafg.de/GRDC/EN/02_srvcs/21_tmsrs/riverdischarge_node.html

    Parameters
    ----------
        station_id: The station id to get. The station id can be found in the
            catalogues at
            https://www.bafg.de/GRDC/EN/02_srvcs/21_tmsrs/212_prjctlgs/project_catalogue_node.html
        start_time: Start time of model in UTC and ISO format string e.g.
            'YYYY-MM-DDTHH:MM:SSZ'.
        end_time: End time of model in  UTC and ISO format string e.g.
            'YYYY-MM-DDTHH:MM:SSZ'.
        parameter: optional. The parameter code to get, e.g. ('Q') discharge,
            cubic meters per second.
        data_home : optional. The directory where the daily grdc data is
            located. If left out will use the grdc_location in the eWaterCycle
            configuration file.
        column: optional. Name of column in dataframe. Default: "streamflow".

    Returns:
        grdc data in a dataframe and metadata.

    Examples:
        .. code-block:: python

            from ewatercycle.observation.grdc import get_grdc_data

            df, meta = get_grdc_data('6335020',
                                    '2000-01-01T00:00Z',
                                    '2001-01-01T00:00Z')
            df.describe()
                     streamflow
            count   4382.000000
            mean    2328.992469
            std	    1190.181058
            min	     881.000000
            25%	    1550.000000
            50%	    2000.000000
            75%	    2730.000000
            max	   11300.000000

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
    if data_home:
        data_path = _to_absolute_path(data_home)
    else:
        raise ValueError(
            "Provide the grdc path using `data_home` argument"
            "or using `grdc_location` in ewatercycle configuration file."
        )

    if not data_path.exists():
        raise ValueError(f"The grdc directory {data_path} does not exist!")

    # Read the raw data
    raw_file = data_path / f"{station_id}_{parameter}_Day.Cmd.txt"
    if not raw_file.exists():
        raise ValueError(f"The grdc file {raw_file} does not exist!")

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


def grdcstationdata2csvandshp(grdc_path, des_dir):
    all_gdfs = []  # List to store all GeoDataFrames
    phone_codes = []
    for grdc_file in os.listdir(grdc_path):
        if grdc_file.endswith(".txt"):
            try:
                df, meta = read_grdc_daily_data(
                    grdc_file.split("_")[0],
                    "1980-01-01T00:00Z",
                    "2001-01-01T00:00Z",
                    grdc_path,
                )

                geometry = [
                    Point(
                        meta["grdc_longitude_in_arc_degree"],
                        meta["grdc_latitude_in_arc_degree"],
                    )
                ]

                country = world[world.geometry.contains(geometry[0])]
                phone_code = country_phone_codes.get(country.iloc[0]["name"], "Unknown")
                phone_codes.append(phone_code)

                # Save the data as a CSV file
                sta_id = str(meta["id_from_grdc"]).zfill(8)
                save_csv_file = f"zq_{phone_code}_grdc{sta_id}.csv"
                save_dir = os.path.join(des_dir, "zq_stations")
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                grdc_file_path = os.path.join(save_dir, save_csv_file)
                df.to_csv(grdc_file_path)

                gdf = gpd.GeoDataFrame([meta], geometry=geometry)
                all_gdfs.append(gdf)  # Append the GeoDataFrame to the list

            except FileNotFoundError as e:
                print(f"Error reading data for station {grdc_file}: {e}")

    # Concatenate all GeoDataFrames into a single GeoDataFrame
    combined_gdf = pd.concat(all_gdfs)

    # Set the coordinate reference system (CRS) to WGS84
    combined_gdf.crs = "EPSG:4326"
    # Define a dictionary where keys are old names and values are new names
    new_column_names = {
        "id_from_grdc": "STCD",
        "station_name": "STNAME",
        "grdc_latitude_in_arc_degree": "LAT",
        "grdc_longitude_in_arc_degree": "LON",
    }
    # Rename the columns
    combined_gdf = combined_gdf.rename(columns=new_column_names)
    # Drop the columns you don't need
    columns_to_drop = [
        col
        for col in combined_gdf.columns
        if col not in new_column_names.values() and col != "geometry"
    ]
    combined_gdf = combined_gdf.drop(columns=columns_to_drop)

    # Add new columns and set their values
    combined_gdf["STCD"] = combined_gdf["STCD"].astype(str).str.zfill(8)
    combined_gdf["ID"] = [
        pc + "_grdc" + sc for pc, sc in zip(phone_codes, combined_gdf["STCD"].values)
    ]
    combined_gdf["STTYPE"] = "streamflow_station"
    combined_gdf["VARTYPE"] = "streamflow"
    combined_gdf["AGENCY"] = np.nan
    combined_gdf["DIVISION"] = np.nan
    combined_gdf["SOURCE"] = "GRDC"

    combined_gdf = combined_gdf[STATION_LST_COLUMNS + ["geometry"]]
    save_zq_dir = os.path.join(des_dir, "stations_list")
    if not os.path.exists(save_zq_dir):
        os.makedirs(save_zq_dir)
    save_shp(combined_gdf, save_zq_dir, "zq_stations")
    # Drop the geometry column and save as CSV
    combined_csv_path = os.path.join(save_zq_dir, "zq_stations.csv")
    combined_gdf.drop(columns=["geometry"]).to_csv(combined_csv_path, index=False)


if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="Give me paths..")

    # Add the arguments
    parser.add_argument(
        "--grdc_path",
        type=str,
        help="The directory where the final ZIP file will be saved",
        default=r"C:\Users\wenyu\.hydrodataset\cache\grdc",
    )
    parser.add_argument(
        "--basin_dir",
        type=str,
        help="The name of the directory to move basin-files to",
        default="D:\\data\\waterism\\basins-origin\\basins_list",
    )
    parser.add_argument(
        "--station_dir",
        type=str,
        help="The name of the directory to upload station-files to",
        default="D:\\data\\waterism\\stations-origin",
    )
    # Parse the arguments
    args = parser.parse_args()
    geojson_path = os.path.join(args.grdc_path, "stationbasins.geojson")
    grdcstationdata2csvandshp(args.grdc_path, args.station_dir)
    onegeojson2basinsshp4grdc(geojson_path, args.basin_dir)
