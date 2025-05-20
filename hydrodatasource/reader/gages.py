"""
Author: Wenyu Ouyang
Date: 2021-12-05 11:21:58
LastEditTime: 2025-05-14 14:13:37
LastEditors: Wenyu Ouyang
Description: Data source class for Gages
FilePath: \hydrodatasource\hydrodatasource\reader\gages.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""

import collections
import logging
import os

import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta
from typing import Tuple, Dict, Union
import pytz
from pandas.core.dtypes.common import is_string_dtype, is_numeric_dtype
from tqdm import tqdm
from hydroutils import hydro_logger, hydro_file, hydro_stat
from hydrodatasource.reader.data_source import HydroData


class Gages(HydroData):
    def __init__(self, data_path, download=False):
        super().__init__(data_path)
        self.data_source_description = self.set_data_source_describe()
        if download:
            self.download_data_source()
        self.gages_sites = self.read_site_info()

    def get_name(self):
        return "GAGES"

    def get_constant_cols(self) -> np.array:
        """all readable attrs in GAGES-II"""
        dir_gage_attr = self.data_source_description["GAGES_ATTR_DIR"]
        var_desc_file = os.path.join(dir_gage_attr, "variable_descriptions.txt")
        var_desc = pd.read_csv(var_desc_file)
        return var_desc["VARIABLE_NAME"].values

    def get_relevant_cols(self):
        return np.array(["dayl", "prcp", "srad", "swe", "tmax", "tmin", "vp"])

    def get_target_cols(self):
        return np.array(["usgsFlow"])

    def get_other_cols(self) -> dict:
        return {
            "FDC": {"time_range": ["1980-01-01", "2000-01-01"], "quantile_num": 100}
        }

    def set_data_source_describe(self):
        gages_db = self.data_source_dir
        # region shapefiles
        gage_region_dir = os.path.join(
            gages_db,
            "boundaries_shapefiles_by_aggeco",
            "boundaries-shapefiles-by-aggeco",
        )
        gages_regions = [
            "bas_ref_all",
            "bas_nonref_CntlPlains",
            "bas_nonref_EastHghlnds",
            "bas_nonref_MxWdShld",
            "bas_nonref_NorthEast",
            "bas_nonref_SECstPlain",
            "bas_nonref_SEPlains",
            "bas_nonref_WestMnts",
            "bas_nonref_WestPlains",
            "bas_nonref_WestXeric",
        ]
        # point shapefile
        gagesii_points_file = os.path.join(
            gages_db, "gagesII_9322_point_shapefile", "gagesII_9322_sept30_2011.shp"
        )

        # config of flow data
        flow_dir = os.path.join(gages_db, "gages_streamflow", "gages_streamflow")
        # forcing
        forcing_dir = os.path.join(gages_db, "basin_mean_forcing", "basin_mean_forcing")
        forcing_types = ["daymet"]
        # attr
        attr_dir = os.path.join(
            gages_db, "basinchar_and_report_sept_2011", "spreadsheets-in-csv-format"
        )
        gauge_id_file = os.path.join(attr_dir, "conterm_basinid.txt")

        download_url_lst = [
            "https://water.usgs.gov/GIS/dsdl/basinchar_and_report_sept_2011.zip",
            "https://water.usgs.gov/GIS/dsdl/gagesII_9322_point_shapefile.zip",
            "https://water.usgs.gov/GIS/dsdl/boundaries_shapefiles_by_aggeco.zip",
            "https://www.sciencebase.gov/catalog/file/get/59692a64e4b0d1f9f05fbd39",
        ]
        usgs_streamflow_url = "https://waterdata.usgs.gov/nwis/dv?cb_00060=on&format=rdb&site_no={}&referred_module=sw&period=&begin_date={}-{}-{}&end_date={}-{}-{}"
        # GAGES-II time series data_source dir
        gagests_dir = os.path.join(gages_db, "59692a64e4b0d1f9f05f")
        population_file = os.path.join(
            gagests_dir,
            "Dataset8_Population-Housing",
            "Dataset8_Population-Housing",
            "PopulationHousing.txt",
        )
        wateruse_file = os.path.join(
            gagests_dir,
            "Dataset10_WaterUse",
            "Dataset10_WaterUse",
            "WaterUse_1985-2010.txt",
        )
        return collections.OrderedDict(
            GAGES_DIR=gages_db,
            GAGES_FLOW_DIR=flow_dir,
            GAGES_FORCING_DIR=forcing_dir,
            GAGES_FORCING_TYPE=forcing_types,
            GAGES_ATTR_DIR=attr_dir,
            GAGES_GAUGE_FILE=gauge_id_file,
            GAGES_DOWNLOAD_URL_LST=download_url_lst,
            GAGES_REGIONS_SHP_DIR=gage_region_dir,
            GAGES_REGION_LIST=gages_regions,
            GAGES_POINT_SHP_FILE=gagesii_points_file,
            GAGES_POPULATION_FILE=population_file,
            GAGES_WATERUSE_FILE=wateruse_file,
            USGS_FLOW_URL=usgs_streamflow_url,
        )

    def read_other_cols(self, object_ids=None, other_cols=None, **kwargs) -> dict:
        # TODO: not finish
        out_dict = {}
        for key, value in other_cols.items():
            if key != "FDC":
                raise NotImplementedError("No this item yet!!")
            assert "time_range" in value.keys()
            if "quantile_num" in value.keys():
                quantile_num = value["quantile_num"]
                out = hydro_stat.cal_fdc(
                    self.read_target_cols(object_ids, value["time_range"], "usgsFlow"),
                    quantile_num=quantile_num,
                )
            else:
                out = hydro_stat.cal_fdc(
                    self.read_target_cols(object_ids, value["time_range"], "usgsFlow")
                )
            out_dict[key] = out
        return out_dict

    def read_attr_all(self, gages_ids: Union[list, np.ndarray]):
        """
        read all attr data for some sites in GAGES-II
        TODO: now it is not same as functions in CAMELS where read_attr_all has no "gages_ids" parameter

        Parameters
        ----------
        gages_ids : Union[list, np.ndarray]
            gages sites' ids

        Returns
        -------
        ndarray
            all attr data for gages_ids
        """
        dir_gage_attr = self.data_source_description["GAGES_ATTR_DIR"]
        f_dict = {}  # factorize dict
        # each key-value pair for atts in a file (list）
        var_dict = {}
        # all attrs
        var_lst = []
        out_lst = []
        key_lst = self._get_attr_col_names(dir_gage_attr)
        for key in key_lst:
            # in "spreadsheets-in-csv-format" directory, the name of "flow_record" file is conterm_flowrec.txt
            if key == "flow_record":
                key = "flowrec"
            data_file = os.path.join(dir_gage_attr, "conterm_" + key + ".txt")
            # remove some unused atttrs in bas_classif
            if key == "bas_classif":
                # https://stackoverflow.com/questions/22216076/unicodedecodeerror-utf8-codec-cant-decode-byte-0xa5-in-position-0-invalid-s
                data_temp = pd.read_csv(
                    data_file,
                    sep=",",
                    dtype={"STAID": str},
                    usecols=range(4),
                    encoding="unicode_escape",
                )
            else:
                data_temp = pd.read_csv(data_file, sep=",", dtype={"STAID": str})
            if key == "flowrec":
                # remove final column which is nan
                data_temp = data_temp.iloc[:, range(data_temp.shape[1] - 1)]
            # all attrs in files
            var_lst_temp = list(data_temp.columns[1:])
            var_dict[key] = var_lst_temp
            var_lst.extend(var_lst_temp)
            k = 0
            n_gage = len(gages_ids)
            out_temp = np.full(
                [n_gage, len(var_lst_temp)], np.nan
            )  # 1d:sites，2d: attrs in current data_file
            # sites intersection，ind2 is the index of sites in conterm_ files，set them in out_temp
            range1 = gages_ids
            range2 = data_temp.iloc[:, 0].astype(str).tolist()
            assert all(x < y for x, y in zip(range2, range2[1:]))
            # Notice the sequence of station ids ! Some id_lst_all are not sorted, so don't use np.intersect1d
            ind2 = [range2.index(tmp) for tmp in range1]
            for field in var_lst_temp:
                if is_string_dtype(data_temp[field]):  # str vars -> categorical vars
                    value, ref = pd.factorize(data_temp.loc[ind2, field], sort=True)
                    out_temp[:, k] = value
                    f_dict[field] = ref.tolist()
                elif is_numeric_dtype(data_temp[field]):
                    out_temp[:, k] = data_temp.loc[ind2, field].values
                k = k + 1
            out_lst.append(out_temp)
        out = np.concatenate(out_lst, 1)
        return out, var_lst, var_dict, f_dict

    def read_constant_cols(
        self, object_ids=None, constant_cols: list = None, **kwargs
    ) -> np.array:
        """
        read some attrs of some sites

        Parameters
        ----------
        object_ids : [type], optional
            sites_ids, by default None
        constant_cols : list, optional
            attrs' names, by default None

        Returns
        -------
        np.array
            attr data for object_ids
        """
        # assert all(x < y for x, y in zip(object_ids, object_ids[1:]))
        attr_all, var_lst_all, var_dict, f_dict = self.read_attr_all(object_ids)
        ind_var = [var_lst_all.index(var) for var in constant_cols]
        return attr_all[:, ind_var]

    def read_attr_origin(self, gages_ids, attr_lst) -> np.ndarray:
        """
        this function read the attrs data in GAGES-II but not transform them to int when they are str

        Parameters
        ----------
        gages_ids : [type]
            [description]
        attr_lst : [type]
            [description]

        Returns
        -------
        np.ndarray
            the first dim is types of attrs, and the second one is sites
        """
        dir_gage_attr = self.data_source_description["GAGES_ATTR_DIR"]
        key_lst = self._get_attr_col_names(dir_gage_attr)
        out_lst = []
        out_lst.extend([] for _ in range(len(attr_lst)))
        range1 = gages_ids
        gage_id_file = self.data_source_description["GAGES_GAUGE_FILE"]
        data_all = pd.read_csv(gage_id_file, sep=",", dtype={0: str})
        range2 = data_all["STAID"].values.tolist()
        assert all(x < y for x, y in zip(range2, range2[1:]))
        # Notice the sequence of station ids ! Some id_lst_all are not sorted, so don't use np.intersect1d
        ind2 = [range2.index(tmp) for tmp in range1]

        for key in key_lst:
            # in "spreadsheets-in-csv-format" directory, the name of "flow_record" file is conterm_flowrec.txt
            if key == "flow_record":
                key = "flowrec"
            data_file = os.path.join(dir_gage_attr, "conterm_" + key + ".txt")
            if key == "bas_classif":
                data_temp = pd.read_csv(
                    data_file,
                    sep=",",
                    dtype={
                        "STAID": str,
                        "WR_REPORT_REMARKS": str,
                        "ADR_CITATION": str,
                        "SCREENING_COMMENTS": str,
                    },
                    engine="python",
                    encoding="unicode_escape",
                )
            elif key == "bound_qa":
                # "DRAIN_SQKM" already exists
                data_temp = pd.read_csv(
                    data_file,
                    sep=",",
                    dtype={"STAID": str},
                    usecols=[
                        "STAID",
                        "BASIN_BOUNDARY_CONFIDENCE",
                        "NWIS_DRAIN_SQKM",
                        "PCT_DIFF_NWIS",
                        "HUC10_CHECK",
                    ],
                )
            else:
                data_temp = pd.read_csv(data_file, sep=",", dtype={"STAID": str})
            if key == "flowrec":
                data_temp = data_temp.iloc[:, range(data_temp.shape[1] - 1)]
            var_lst_temp = list(data_temp.columns[1:])
            do_exist, idx_lst = is_any_elem_in_a_lst(
                attr_lst, var_lst_temp, return_index=True
            )
            if do_exist:
                for idx in idx_lst:
                    idx_in_var = (
                        var_lst_temp.index(attr_lst[idx]) + 1
                    )  # +1 because the first col of data_temp is ID
                    out_lst[idx] = data_temp.iloc[ind2, idx_in_var].values
            else:
                continue
        return np.array(out_lst)

    def _get_attr_col_names(self, dir_gage_attr):
        var_des = pd.read_csv(
            os.path.join(dir_gage_attr, "variable_descriptions.txt"), sep=","
        )
        var_des_map_values = var_des["VARIABLE_TYPE"].tolist()
        for i in range(len(var_des)):
            var_des_map_values[i] = var_des_map_values[i].lower()
        result = list(set(var_des_map_values))
        result.sort(key=var_des_map_values.index)
        result.remove("x_region_names")
        return result

    def read_forcing_gage(self, usgs_id, var_lst, t_range_list, forcing_type="daymet"):
        gage_dict = self.gages_sites
        ind = np.argwhere(gage_dict["STAID"] == usgs_id)[0][0]
        huc = gage_dict["HUC02"][ind]

        data_folder = os.path.join(
            self.data_source_description["GAGES_FORCING_DIR"], forcing_type
        )
        # original daymet file not for leap year, there is no data in 12.31 in leap year,
        # so files which have been interpolated for nan value have name "_leap"
        data_file = os.path.join(
            data_folder, huc, f"{usgs_id}_lump_{forcing_type}_forcing_leap.txt"
        )
        data_temp = pd.read_csv(data_file, sep=r"\s+", header=None, skiprows=1)

        df_date = data_temp[[0, 1, 2]]
        df_date.columns = ["year", "month", "day"]
        date = pd.to_datetime(df_date).values.astype("datetime64[D]")
        nf = len(var_lst)
        assert all(x < y for x, y in zip(date, date[1:]))
        [c, ind1, ind2] = np.intersect1d(date, t_range_list, return_indices=True)
        assert date[0] <= t_range_list[0] and date[-1] >= t_range_list[-1]
        nt = t_range_list.size
        out = np.empty([nt, nf])
        var_lst_in_file = [
            "dayl(s)",
            "prcp(mm/day)",
            "srad(W/m2)",
            "swe(mm)",
            "tmax(C)",
            "tmin(C)",
            "vp(Pa)",
        ]
        for k in range(nf):
            # assume all files are of same columns. May check later.
            ind = [
                i
                for i in range(len(var_lst_in_file))
                if var_lst[k] in var_lst_in_file[i]
            ][0]
            out[ind2, k] = data_temp[ind + 4].values[ind1]
        return out

    def read_relevant_cols(
        self, object_ids=None, t_range_list=None, var_lst=None, **kwargs
    ) -> np.array:
        assert all(x < y for x, y in zip(object_ids, object_ids[1:]))
        assert all(x < y for x, y in zip(t_range_list, t_range_list[1:]))
        t_lst = hydro_utils.t_range_days(t_range_list)
        nt = t_lst.shape[0]
        x = np.empty([len(object_ids), nt, len(var_lst)])
        for k in tqdm(range(len(object_ids)), desc="reading GAGES forcing data"):
            data = self.read_forcing_gage(
                object_ids[k],
                var_lst,
                t_lst,
                forcing_type=self.data_source_description["GAGES_FORCING_TYPE"][0],
            )
            x[k, :, :] = data
        return x

    def read_target_cols(
        self, usgs_id_lst=None, t_range_list=None, target_cols=None, **kwargs
    ) -> np.array:
        """
        Read USGS daily average streamflow data according to id and time

        Parameters
        ----------
        usgs_id_lst
            site information
        t_range_list
            must be time range for downloaded data
        target_cols

        kwargs
            optional

        Returns
        -------
        np.array
            streamflow data, 1d-axis: gages, 2d-axis: day, 3d-axis: streamflow
        """
        t_lst = hydro_utils.t_range_days(t_range_list)
        nt = t_lst.shape[0]
        y = np.empty([len(usgs_id_lst), nt, 1])
        for k in tqdm(range(len(usgs_id_lst)), desc="Read GAGES streamflow data"):
            data_obs = self.read_usgs_gage(usgs_id_lst[k], t_lst)
            y[k, :, 0] = data_obs
        return y

    def read_usgs_gage(self, usgs_id, t_lst):
        """
        read data for one gage

        Parameters
        ----------
        usgs_id : [type]
            [description]
        t_lst : [type]
            [description]

        Returns
        -------
        [type]
            [description]
        """
        dir_gage_flow = self.data_source_description["GAGES_FLOW_DIR"]
        gage_id_df = pd.DataFrame(self.gages_sites)
        huc = gage_id_df[gage_id_df["STAID"] == usgs_id]["HUC02"].values[0]
        usgs_file = os.path.join(dir_gage_flow, str(huc), usgs_id + ".txt")
        # ignore the comment lines and the first non-value row
        df_flow = pd.read_csv(
            usgs_file, comment="#", sep="\t", dtype={"site_no": str}
        ).iloc[1:, :]
        # reset the index, start from 0
        df_flow = df_flow.reset_index(drop=True)
        # change the original column names
        columns_names = df_flow.columns.tolist()
        columns_flow = []
        columns_flow.extend(
            column_name
            for column_name in columns_names
            if "_00060_00003" in column_name and "_00060_00003_cd" not in column_name
        )
        columns_flow_cd = [
            column_name
            for column_name in columns_names
            if "_00060_00003_cd" in column_name
        ]
        if len(columns_flow) > 1:
            self._chose_flow_col(df_flow, t_lst, columns_flow, columns_flow_cd)
        else:
            for column_name in columns_names:
                if (
                    "_00060_00003" in column_name
                    and "_00060_00003_cd" not in column_name
                ):
                    df_flow.rename(columns={column_name: "flow"}, inplace=True)
                    break
            for column_name in columns_names:
                if "_00060_00003_cd" in column_name:
                    df_flow.rename(columns={column_name: "mode"}, inplace=True)
                    break

        columns = ["agency_cd", "site_no", "datetime", "flow", "mode"]
        if df_flow.empty:
            df_flow = pd.DataFrame(columns=columns)
        if "flow" not in df_flow.columns.intersection(columns):
            data_temp = df_flow.loc[:, df_flow.columns.intersection(columns)]
            # add nan column to data_temp
            data_temp = pd.concat([data_temp, pd.DataFrame(columns=["flow", "mode"])])
        else:
            data_temp = df_flow.loc[:, columns]
        self._set_some_value_to_nan(data_temp, "flow")
        # set negative value -- nan
        obs = data_temp["flow"].astype("float").values
        obs[obs < 0] = np.nan
        # time range intersection. set points without data nan values
        nt = len(t_lst)
        out = np.full([nt], np.nan)
        # date in df is str，so transform them to datetime
        df_date = data_temp["datetime"]
        date = pd.to_datetime(df_date).values.astype("datetime64[D]")
        c, ind1, ind2 = np.intersect1d(date, t_lst, return_indices=True)
        out[ind2] = obs[ind1]
        return out

    def _chose_flow_col(self, df_flow, t_lst, columns_flow, columns_flow_cd):
        logging.debug("there are some columns for flow, choose one\n")
        df_date_temp = df_flow["datetime"]
        date_temp = pd.to_datetime(df_date_temp).values.astype("datetime64[D]")
        c_temp, ind1_temp, ind2_temp = np.intersect1d(
            date_temp, t_lst, return_indices=True
        )
        num_nan_lst = []
        for item in columns_flow:
            out_temp = np.full([len(t_lst)], np.nan)

            self._set_some_value_to_nan(df_flow, item)
            df_flow_temp = df_flow[item].copy()
            out_temp[ind2_temp] = df_flow_temp[ind1_temp]
            num_nan = np.isnan(out_temp).sum()
            num_nan_lst.append(num_nan)
        num_nan_np = np.array(num_nan_lst)
        index_flow_num = np.argmin(num_nan_np)
        df_flow.rename(columns={columns_flow[index_flow_num]: "flow"}, inplace=True)
        df_flow.rename(columns={columns_flow_cd[index_flow_num]: "mode"}, inplace=True)

    def _set_some_value_to_nan(self, arg0, arg1):
        arg0.loc[arg0[arg1] == "Ice", arg1] = np.nan
        arg0.loc[arg0[arg1] == "Ssn", arg1] = np.nan
        arg0.loc[arg0[arg1] == "Tst", arg1] = np.nan
        arg0.loc[arg0[arg1] == "Eqp", arg1] = np.nan
        arg0.loc[arg0[arg1] == "Rat", arg1] = np.nan
        arg0.loc[arg0[arg1] == "Dis", arg1] = np.nan
        arg0.loc[arg0[arg1] == "Bkw", arg1] = np.nan
        arg0.loc[arg0[arg1] == "***", arg1] = np.nan
        arg0.loc[arg0[arg1] == "Mnt", arg1] = np.nan
        arg0.loc[arg0[arg1] == "ZFL", arg1] = np.nan

    def read_object_ids(self, object_params=None) -> np.array:
        return self.gages_sites["STAID"]

    def read_basin_area(self, object_ids) -> np.array:
        return self.read_constant_cols(object_ids, ["DRAIN_SQKM"], is_return_dict=False)

    def read_mean_prep(self, object_ids) -> np.array:
        mean_prep = self.read_constant_cols(
            object_ids, ["PPTAVG_BASIN"], is_return_dict=False
        )
        mean_prep = mean_prep / 365 * 10
        return mean_prep

    def download_data_source(self):
        print("Please download data manually!")
        if not os.path.isdir(self.data_source_description["GAGES_DIR"]):
            os.makedirs(self.data_source_description["GAGES_DIR"])
        zip_files = [
            "59692a64e4b0d1f9f05fbd39",
            "basin_mean_forcing.zip",
            "basinchar_and_report_sept_2011.zip",
            "boundaries_shapefiles_by_aggeco.zip",
            "gages_streamflow.zip",
            "gagesII_9322_point_shapefile.zip",
        ]
        download_zip_files = [
            os.path.join(self.data_source_description["GAGES_DIR"], zip_file)
            for zip_file in zip_files
        ]
        for download_zip_file in download_zip_files:
            if not os.path.isfile(download_zip_file):
                raise RuntimeError(
                    download_zip_file + " not found! Please download the data"
                )
        unzip_dirs = [
            os.path.join(self.data_source_description["GAGES_DIR"], zip_file[:-4])
            for zip_file in zip_files
        ]
        for i in range(len(unzip_dirs)):
            if not os.path.isdir(unzip_dirs[i]):
                print("unzip directory:" + unzip_dirs[i])
                unzip_nested_zip(download_zip_files[i], unzip_dirs[i])
            else:
                print("unzip directory -- " + unzip_dirs[i] + " has existed")

    def read_site_info(self):
        gage_id_file = self.data_source_description["GAGES_GAUGE_FILE"]
        data_all = pd.read_csv(gage_id_file, sep=",", dtype={0: str})
        gage_fld_lst = data_all.columns.values
        out = {}
        df_id_region = data_all.iloc[:, 0].values
        assert all(x < y for x, y in zip(df_id_region, df_id_region[1:]))
        for s in gage_fld_lst:
            if s is gage_fld_lst[1]:
                out[s] = data_all[s].values.tolist()
            else:
                out[s] = data_all[s].values
        return pd.DataFrame(out)

    def prepare_usgs_data(self):
        hydro_logger.info(
            "NOT all data_source could be downloaded from website directly!"
        )
        data_source_description = self.data_source_description
        # download zip files
        [
            hydro_file.download_one_zip(attr_url, data_source_description["GAGES_DIR"])
            for attr_url in data_source_description["GAGES_DOWNLOAD_URL_LST"]
        ]
        # download streamflow data from USGS website
        dir_gage_flow = data_source_description["GAGES_FLOW_DIR"]
        streamflow_url = data_source_description["USGS_FLOW_URL"]
        # TODO: now a hard code for t_download_range
        t_download_range = ["1980-01-01", "2025-01-01"]
        if not os.path.isdir(dir_gage_flow):
            os.makedirs(dir_gage_flow)
        dir_list = os.listdir(dir_gage_flow)
        # if no streamflow data for the usgs_id_lst, then download them from the USGS website
        data_all = pd.read_csv(
            data_source_description["GAGES_GAUGE_FILE"], sep=",", dtype={0: str}
        )
        usgs_id_lst = data_all.iloc[:, 0].values.tolist()
        gage_fld_lst = data_all.columns.values
        for ind in range(len(usgs_id_lst)):  # different hucs different directories
            huc_02 = data_all[gage_fld_lst[3]][ind]
            dir_huc_02 = str(huc_02)
            if dir_huc_02 not in dir_list:
                dir_huc_02 = os.path.join(dir_gage_flow, str(huc_02))
                os.mkdir(dir_huc_02)
                dir_list = os.listdir(dir_gage_flow)
            dir_huc_02 = os.path.join(dir_gage_flow, str(huc_02))
            file_list = os.listdir(dir_huc_02)
            file_usgs_id = f"{str(usgs_id_lst[ind])}.txt"
            if file_usgs_id not in file_list:
                # download data and save as txt file
                start_time_str = datetime.strptime(t_download_range[0], "%Y-%m-%d")
                end_time_str = datetime.strptime(
                    t_download_range[1], "%Y-%m-%d"
                ) - timedelta(days=1)
                url = streamflow_url.format(
                    usgs_id_lst[ind],
                    start_time_str.year,
                    start_time_str.month,
                    start_time_str.day,
                    end_time_str.year,
                    end_time_str.month,
                    end_time_str.day,
                )

                # save in its HUC02 dir
                temp_file = os.path.join(dir_huc_02, f"{str(usgs_id_lst[ind])}.txt")
                hydro_file.download_small_file(url, temp_file)
                print("successfully download " + temp_file + " streamflow data!")

    def cache_attributes_xrdataset(self):
        """Convert all the attributes to a single dataset

        Returns
        -------
        None
        """
        # NOTICE: although it seems that we don't use pint_xarray, we have to import this package
        import pint_xarray  # noqa: F401

        shape_dir = os.path.join(
            self.data_source_description["SHAPE_DIR"], "basins.shp"
        )
        if "s3://" in shape_dir:
            with conf.FS.open(shape_dir, mode="rb") as f:
                shape = gpd.read_file(f)
        else:
            shape = gpd.read_file(shape_dir)
        df_area = cal_area_from_shp(shape)  # calculate the area from shape file
        df_area.set_index("basin_id", inplace=True)

        df_attr = self.read_attributes()
        df_attr.set_index("basin_id", inplace=True)
        df_attr = df_attr.join(df_area)
        # Mapping provided units to the variables in the datasets
        # For attributes, all the variables' units are same in all unit_info files
        # hence, we just chose the first one
        if "s3://" in self.data_source_description["UNIT_FILES"][0]:
            with conf.FS.open(
                self.data_source_description["UNIT_FILES"][0], mode="rb"
            ) as fp:
                units_dict = json.load(fp)
        else:
            units_dict = hydro_file.unserialize_json(
                self.data_source_description["UNIT_FILES"][0]
            )
        units_dict["shp_area"] = "km^2"  # add the unit of shp_area
        # Convert string columns to categorical variables and record categorical mappings
        categorical_mappings = {}
        for column in df_attr.columns:
            if df_attr[column].dtype == "object":
                df_attr[column] = df_attr[column].astype("category")
                categorical_mappings[column] = dict(
                    enumerate(df_attr[column].cat.categories)
                )
                df_attr[column] = df_attr[column].cat.codes

        ds = xr.Dataset()
        for column in df_attr.columns:
            attrs = {"units": units_dict.get(column, "unknown")}
            if column in categorical_mappings:
                attrs["category_mapping"] = categorical_mappings[column]

            data_array = xr.DataArray(
                data=df_attr[column].values,
                dims=["basin"],
                # we have set gage_id as index so that it won't be saved as numeric values
                coords={"basin": df_attr.index.values.astype(str)},
                attrs=attrs,
            )
            ds[column] = data_array

        # Convert categorical mappings to strings
        for column in ds.data_vars:
            if "category_mapping" in ds[column].attrs:
                # Convert the dictionary to a string
                mapping_str = str(ds[column].attrs["category_mapping"])
                ds[column].attrs["category_mapping"] = mapping_str
        dataset_name = self.dataset_name
        prefix_ = "" if dataset_name is None else dataset_name + "_"
        ds.to_netcdf(os.path.join(CACHE_DIR, f"{prefix_}attributes.nc"))

    def cache_timeseries_xrdataset(self, t_range=None, **kwargs):
        """Save all timeseries data in separate NetCDF files for each time unit.

        Parameters
        ----------
        t_range : list, optional
            Time range for the data, by default ["1980-01-01", "2023-12-31"]
        kwargs : dict, optional
            batchsize -- Number of basins to process per batch, by default 100
            time_units -- List of time units to process, by default None
            start0101_freq -- for freq setting, if the start date is 01-01, set True, by default False
        """
        batchsize = kwargs.get("batchsize", 100)
        time_units = kwargs.get("time_units", self.time_unit) or [
            "1D"
        ]  # Default to ["1D"] if not specified or if time_units is None
        start0101_freq = kwargs.get("start0101_freq", False)

        variables = self.get_timeseries_cols()
        basins = self.camels_sites["basin_id"].values

        # Define the generator function for batching
        def data_generator(basins, batch_size):
            for i in range(0, len(basins), batch_size):
                yield basins[i : i + batch_size]

        for time_unit in time_units:
            if t_range is None:
                if time_unit != "3h":
                    t_range = ["1980-01-01", "2023-12-31"]
                else:
                    t_range = ["1980-01-01 01", "2023-12-31 22"]

            # Generate the time range specific to the time unit
            if start0101_freq:
                times = (
                    generate_start0101_time_range(
                        start_time=t_range[0], end_time=t_range[-1], freq=time_unit
                    )
                    .strftime("%Y-%m-%d %H:%M:%S")
                    .tolist()
                )
            else:
                times = (
                    pd.date_range(start=t_range[0], end=t_range[-1], freq=time_unit)
                    .strftime("%Y-%m-%d %H:%M:%S")
                    .tolist()
                )
            # Retrieve the correct units information for this time unit
            unit_file = next(
                file
                for file in self.data_source_description["UNIT_FILES"]
                if time_unit in file
            )
            if "s3://" in unit_file:
                with conf.FS.open(unit_file, mode="rb") as fp:
                    units_info = json.load(fp)
            else:
                units_info = hydro_file.unserialize_json(unit_file)

            for basin_batch in data_generator(basins, batchsize):
                data = self.read_timeseries(
                    object_ids=basin_batch,
                    t_range_list=t_range,
                    relevant_cols=variables[
                        time_unit
                    ],  # Ensure we use the right columns for the time unit
                    time_units=[
                        time_unit
                    ],  # Pass the time unit to ensure correct data retrieval
                    start0101_freq=start0101_freq,
                )

                dataset = xr.Dataset(
                    data_vars={
                        variables[time_unit][i]: (
                            ["basin", "time"],
                            data[time_unit][:, :, i],
                            {"units": units_info[variables[time_unit][i]]},
                        )
                        for i in range(len(variables[time_unit]))
                    },
                    coords={
                        "basin": basin_batch,
                        "time": pd.to_datetime(times),
                    },
                )

                # Save the dataset to a NetCDF file for the current batch and time unit
                prefix_ = self._get_ts_file_prefix_(self.dataset_name, self.version)
                batch_file_path = os.path.join(
                    CACHE_DIR,
                    f"{prefix_}timeseries_{time_unit}_batch_{basin_batch[0]}_{basin_batch[-1]}.nc",
                )
                dataset.to_netcdf(batch_file_path)

                # Release memory by deleting the dataset
                del dataset
                del data

    def cache_xrdataset(self, t_range=None, time_units=None):
        """Save all data in a netcdf file in the cache directory"""
        self.cache_attributes_xrdataset()
        self.cache_timeseries_xrdataset(t_range=t_range, time_units=time_units)

    def read_ts_xrdataset(
        self,
        gage_id_lst: list = None,
        t_range: list = None,
        var_lst: list = None,
        **kwargs,
    ) -> dict:
        """
        Read time-series xarray dataset from multiple NetCDF files and organize them by time units.

        Parameters:
        ----------
        gage_id_lst: list
            List of gage IDs to select.
        t_range: list
            List of two elements [start_time, end_time] to select time range.
        var_lst: list
            List of variables to select.
        **kwargs
            Additional arguments.

        Returns:
        ----------
        dict: A dictionary where each key is a time unit and each value is an xarray.Dataset containing the selected gage IDs, time range, and variables.
        """
        dataset_name = self.dataset_name
        version = self.version
        time_units = kwargs.get("time_units", self.time_unit)
        if var_lst is None:
            return None

        # Initialize a dictionary to hold datasets for each time unit
        datasets_by_time_unit = {}

        prefix_ = self._get_ts_file_prefix_(dataset_name, version)

        for time_unit in time_units:
            # Collect batch files specific to the current time unit
            batch_files = self._get_batch_files(prefix_, time_unit)

            if not batch_files:
                # Cache the data if no batch files are found for the current time unit
                self.cache_timeseries_xrdataset(**kwargs)
                batch_files = self._get_batch_files(prefix_, time_unit)

            selected_datasets = []

            for batch_file in batch_files:
                ds = xr.open_dataset(batch_file)
                all_vars = ds.data_vars
                if any(var not in ds.variables for var in var_lst):
                    raise ValueError(f"var_lst must all be in {all_vars}")
                if valid_gage_ids := [
                    gid for gid in gage_id_lst if gid in ds["basin"].values
                ]:
                    ds_selected = ds[var_lst].sel(
                        basin=valid_gage_ids, time=slice(t_range[0], t_range[1])
                    )
                    selected_datasets.append(ds_selected)

                ds.close()  # Close the dataset to free memory

            # If any datasets were selected, concatenate them along the 'basin' dimension
            if selected_datasets:
                # NOTE: the chosen part must be sorted by basin, or there will be some negative sideeffect for continue usage of this repo
                datasets_by_time_unit[time_unit] = xr.concat(
                    selected_datasets, dim="basin"
                ).sortby("basin")
            else:
                datasets_by_time_unit[time_unit] = xr.Dataset()

        return datasets_by_time_unit

    def _get_ts_file_prefix_(self, dataset_name, version):
        prefix_ = "" if dataset_name is None else dataset_name + "_"
        # we add version for prefix_ as we will update the dataset iteratively
        prefix_ = prefix_ + f"{version}_" if version is not None else prefix_
        return prefix_

    def _get_batch_files(self, prefix_, time_unit):
        return [
            os.path.join(CACHE_DIR, f)
            for f in os.listdir(CACHE_DIR)
            if re.match(
                rf"^{prefix_}timeseries_{time_unit}_batch_[A-Za-z0-9_]+_[A-Za-z0-9_]+\.nc$",
                f,
            )
        ]

    def read_attr_xrdataset(self, gage_id_lst=None, var_lst=None, **kwargs):
        dataset_name = self.dataset_name

        prefix_ = "" if dataset_name is None else dataset_name + "_"
        if var_lst is None or len(var_lst) == 0:
            return None
        try:
            attr = xr.open_dataset(os.path.join(CACHE_DIR, f"{prefix_}attributes.nc"))
        except FileNotFoundError:
            self.cache_attributes_xrdataset()
            attr = xr.open_dataset(os.path.join(CACHE_DIR, f"{prefix_}attributes.nc"))
        return attr[var_lst].sel(basin=gage_id_lst)


def get_dor_values(gages: Gages, usgs_id) -> np.array:
    """
    get dor values from gages for the usgs_id-sites

    """

    assert all(x < y for x, y in zip(usgs_id, usgs_id[1:]))
    # mm/year 1-km grid,  megaliters total storage per sq km  (1 megaliters = 1,000,000 liters = 1,000 cubic meters)
    # attr_lst = ["RUNAVE7100", "STOR_NID_2009"]
    attr_lst = ["RUNAVE7100", "STOR_NOR_2009"]
    data_attr = gages.read_constant_cols(usgs_id, attr_lst)
    run_avg = data_attr[:, 0] * (10 ** (-3)) * (10**6)  # m^3 per year
    nor_storage = data_attr[:, 1] * 1000  # m^3
    return nor_storage / run_avg


def get_diversion(gages: Gages, usgs_id) -> np.array:
    diversion_strs = ["diversion", "divert"]
    assert all(x < y for x, y in zip(usgs_id, usgs_id[1:]))
    attr_lst = ["WR_REPORT_REMARKS", "SCREENING_COMMENTS"]
    data_attr = gages.read_attr_origin(usgs_id, attr_lst)
    diversion_strs_lower = [elem.lower() for elem in diversion_strs]
    data_attr0_lower = np.array(
        [elem.lower() if type(elem) == str else elem for elem in data_attr[0]]
    )
    data_attr1_lower = np.array(
        [elem.lower() if type(elem) == str else elem for elem in data_attr[1]]
    )
    data_attr_lower = np.vstack((data_attr0_lower, data_attr1_lower)).T
    diversions = [
        is_any_elem_in_a_lst(diversion_strs_lower, data_attr_lower[i], include=True)
        for i in range(len(usgs_id))
    ]
    return np.array(diversions)


def is_any_elem_in_a_lst(lst1, lst2, return_index=False, include=False):
    do_exist = False
    idx_lst = []
    for j in range(len(lst1)):
        if include:
            for lst2_elem in lst2:
                if lst1[j] in lst2_elem:
                    idx_lst.append(j)
                    do_exist = True
        elif lst1[j] in lst2:
            idx_lst.append(j)
            do_exist = True
    return (do_exist, idx_lst) if return_index else do_exist
