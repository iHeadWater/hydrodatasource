import collections
import os
from abc import ABC
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import xarray as xr
from hydrodataset import HydroDataset
from hydroutils import hydro_file, hydro_time
from tqdm import tqdm

import hydrodatasource.configs.config as conf
from hydrodatasource.configs.config import SETTING
from hydrodatasource.reader import access_fs
from hydrodatasource.reader.reader import DataHandler

CACHE_DIR = hydro_file.get_cache_dir()


class HydroData(ABC):
    """An interface for reading multi-modal data sources.

    Parameters
    ----------
    ABC : _type_
        _description_
    """

    def __init__(self, data_path):
        self.data_source_dir = data_path

    def get_name(self):
        raise NotImplementedError

    def set_data_source_describe(self):
        raise NotImplementedError

    def read_data(self):
        raise NotImplementedError


class SelfMadeCamels(HydroDataset):
    """A class for reading hydrodataset, but not really ready-datasets,
    just some data directorys organized like a CAMELS dataset.

    Typically, we read our self-made data.

    NOTE:
    open data: Mainly for baisn from GAGES-II (FOR US) or GRDC (FOR CHINA)
    private data: for stations with streamflow data

    No matter open or private, we will compile forcing data and attr data into a dataset,
    organized like a ready dataset -- like CAMELS, as these are for paper publication.

    The process of compiling forcing data and attr data into these streamflow dataset is
    not here. It is in some preprocessing scripts such as CatchmentAttributes repo: https://github.com/OuyangWenyu/CatchmentAttributes
    or CatchmentForcing repo: https://github.com/OuyangWenyu/CatchmentForcings.

    They are generally not full automatic, as some data are not open, and some data are not ready for use.

    For streamflow data, we will transform them to netcdf format, and combine them with forcing data into a timesereis dataset.

    Hence, in this module, we directly read the compiled dataset: one time-series nc file and one attr nc file.
    If any data is not ready, we will raise an error and preprocess in other scripts again.
    """

    def __init__(self, data_path, download=False, version="latest"):
        """Initialize a self-made CAMELS dataset.

        Parameters
        ----------
        data_path : _type_
            _description_
        download : bool, optional
            _description_, by default False
        version : str, optional
            We have multiple versions of self-made CAMELS dataset, by default "latest"
        """
        super().__init__(data_path)
        # the naming convention for basin ids are needed
        # we use GRDC station's ids as our default coding convention
        # GRDC station ids are 7 digits, the first 1 digit is continent code,
        # the second 4 digits are sub-region related code
        self.data_source_description = self.set_data_source_describe()
        if download:
            self.download_data_source()
        self.camels_sites = self.read_site_info()
        # for camels_cc (version 1), {"time": "DATE", "streamflow": "Q"}
        # Here the dict is for camels_cc_v2
        if version in ["latest", "v2"]:
            self.VAR_DICT = {"time": "time", "streamflow": "streamflow"}
        elif version == "v1":
            self.VAR_DICT = {"time": "DATE", "streamflow": "Q"}
        else:
            raise ValueError("version must be latest, v2, or v1")

    @property
    def streamflow_unit(self):
        return "m^3/s"

    def get_name(self):
        return "SelfMadeCamels"

    def set_data_source_describe(self):
        camels_db = self.data_source_dir
        # shp files of basins
        camels_shp_files_dir = os.path.join(camels_db, "basin_boudaries")
        # attr, flow and forcing data are all in the same dir. each basin has one dir.
        flow_dir = os.path.join(camels_db, "streamflow")
        sm_dir = os.path.join(camels_db, "soil_moisture")
        et_dir = os.path.join(camels_db, "evapotranspiration")
        forcing_dir = os.path.join(camels_db, "basin_mean_forcing")
        attr_dir = os.path.join(camels_db, "attribute")
        # no gauge id file for CAMELS_CC, just read from any attribute file
        gauge_id_file = os.path.join(camels_db, "gage_points.csv")
        attr_key_lst = [
            "climate",
            "geology",
            "land_cover",
            "permeability_porosity",
            "root_depth",
            "soil",
            "topo_elev_slope",
            "topo_shape_factors",
        ]
        return collections.OrderedDict(
            CAMELS_DIR=camels_db,
            CAMELS_FLOW_DIR=flow_dir,
            CAMELS_SM_DIR=sm_dir,
            CAMELS_ET_DIR=et_dir,
            CAMELS_FORCING_DIR=forcing_dir,
            CAMELS_ATTR_DIR=attr_dir,
            CAMELS_ATTR_KEY_LST=attr_key_lst,
            CAMELS_GAUGE_FILE=gauge_id_file,
            CAMELS_BASINS_SHP_DIR=camels_shp_files_dir,
        )

    def download_data_source(self):
        print(
            "Please download it manually and put all files of a CAMELS dataset in the CAMELS_DIR directory."
        )
        print("We unzip all files now.")

    def read_site_info(self):
        camels_file = self.data_source_description["CAMELS_GAUGE_FILE"]
        return pd.read_csv(camels_file, sep=",", dtype={"gage_id": str})

    def read_object_ids(self, object_params=None) -> np.array:
        return self.camels_sites["gage_id"].values

    def read_target_cols(
        self, object_ids=None, t_range_list=None, target_cols=None, **kwargs
    ) -> np.array:
        if target_cols is None:
            return np.array([])
        else:
            nf = len(target_cols)
        t_range_list = hydro_time.t_range_days(t_range_list)
        nt = t_range_list.shape[0]
        y = np.full([len(object_ids), nt, nf], np.nan)
        streamflow_name = self.VAR_DICT["streamflow"]
        time_name = self.VAR_DICT["time"]
        for j in tqdm(
            range(len(target_cols)), desc="Read streamflow data of CAMELS-CC"
        ):
            for k in tqdm(range(len(object_ids))):
                # only one streamflow type: Q
                flow_file = os.path.join(
                    self.data_source_description["CAMELS_FLOW_DIR"],
                    object_ids[k] + ".csv",
                )
                flow_data = pd.read_csv(flow_file, sep=",")
                date = pd.to_datetime(flow_data[time_name]).values.astype(
                    "datetime64[D]"
                )
                [c, ind1, ind2] = np.intersect1d(
                    date, t_range_list, return_indices=True
                )
                y[k, ind2, j] = flow_data[streamflow_name].values[ind1]
        return y

    def read_relevant_cols(
        self, object_ids=None, t_range_list: list = None, relevant_cols=None, **kwargs
    ) -> Union[np.array, list]:
        """3d data (site_num * time_length * var_num), time-series data"""
        t_range_list = hydro_time.t_range_days(t_range_list)
        nt = t_range_list.shape[0]
        x = np.full([len(object_ids), nt, len(relevant_cols)], np.nan)
        for k in tqdm(range(len(object_ids)), desc="Read forcing data of CAMELS-CC"):
            forcing_file = os.path.join(
                self.data_source_description["CAMELS_FORCING_DIR"],
                object_ids[k] + "_lump_era5_land_forcing.txt",
            )
            forcing_data = pd.read_csv(forcing_file, sep=" ")
            df_date = forcing_data[["Year", "Mnth", "Day"]]
            df_date.columns = ["year", "month", "day"]
            date = pd.to_datetime(df_date).values.astype("datetime64[D]")

            [c, ind1, ind2] = np.intersect1d(date, t_range_list, return_indices=True)
            for j in range(len(relevant_cols)):
                if "evaporation" in relevant_cols[j]:
                    # evaporation value are all negative (maybe upward flux is marked as negative)
                    x[k, ind2, j] = (
                        forcing_data[relevant_cols[j]].values[ind1] * -1 * 1e3
                    )
                # unit of prep and pet is m, tran them to mm
                elif "precipitation" in relevant_cols[j]:
                    prcp = forcing_data[relevant_cols[j]].values
                    # there are a few negative values for prcp, set them 0
                    prcp[prcp < 0] = 0.0
                    x[k, ind2, j] = prcp[ind1] * 1e3
                else:
                    x[k, ind2, j] = forcing_data[relevant_cols[j]].values[ind1]
        return x

    def read_constant_cols(
        self, object_ids=None, constant_cols=None, **kwargs
    ) -> np.array:
        """2d data (site_num * var_num), non-time-series data"""
        raise NotImplementedError(
            "Directly read attributes from nc file by read_attr_xrdataset."
        )

    def get_constant_cols(self) -> np.array:
        """the constant cols in this data_source"""
        data_folder = self.data_source_description["CAMELS_ATTR_DIR"]
        files = np.sort(os.listdir(data_folder))
        attr_types = []
        for file_ in files:
            file = os.path.join(data_folder, file_)
            attr_tmp = pd.read_csv(file, sep=",", dtype={"gage_id": str})
            attr_types = attr_types + attr_tmp.columns[1:].values.tolist()
        return np.array(attr_types)

    def get_relevant_cols(self) -> np.array:
        """the relevant cols in this data_source"""
        forcing_dir = self.data_source_description["CAMELS_FORCING_DIR"]
        forcing_file = os.path.join(forcing_dir, os.listdir(forcing_dir)[0])
        forcing_tmp = pd.read_csv(forcing_file, sep="\s+", dtype={"gage_id": str})
        return forcing_tmp.columns.values[4:]

    def get_target_cols(self) -> np.array:
        """the target cols in this data_source"""
        return np.array(["streamflow"])

    def cache_attributes_xrdataset(self):
        """Convert all the attributes to a single dataframe

        Returns
        -------
        None
        """
        # NOTICE: although it seems that we don't use pint_xarray, we have to import this package
        import pint_xarray  # noqa: F401

        attr_files = Path(self.data_source_description["CAMELS_ATTR_DIR"]).glob("*.csv")
        dataframes = {}
        for file_path in attr_files:
            df = pd.read_csv(file_path, index_col="gage_id")
            dataframes[file_path.stem] = df  # use stem as key

        # merged all data
        merged_data = pd.DataFrame()
        for df in dataframes.values():
            if merged_data.empty:
                merged_data = df
            else:
                merged_data = pd.merge(merged_data, df, on="gage_id", how="outer")

        # Mapping provided units to the variables in the datasets
        units_dict = {
            "p_mean": "mm/day",
            "pet_mean": "mm/day",
            "aet_mean": "mm/day",
            "aridity": "dimensionless",
            "p_seasonality": "dimensionless",
            "high_prec_freq": "days/year",
            "high_prec_dur": "day",
            "high_prec_timing": "dimensionless",
            "low_prec_freq": "days/year",
            "low_prec_dur": "day",
            "low_prec_timing": "dimensionless",
            "frac_snow_daily": "dimensionless",
            # Adding some additional units for other variables based on the provided units dict
            "elev": "m",
            "slope": "m/km",
            # Geology data
            "geol_class_1st": "dimensionless",
            "geol_class_1st_frac": "dimensionless",
            "geol_class_2nd": "dimensionless",
            "geol_class_2nd_frac": "dimensionless",
            "carb_rocks_frac": "dimensionless",
            # Land Cover
            "EvergreenNeedleleafTree": "dimensionless",
            "EvergreenBroadleafTree": "dimensionless",
            # ... other land cover variables ...
            # Permeability and Porosity
            "Permeability": "m^2",
            "Porosity": "dimensionless",
            # Root Depth
            "root_depth_50": "m",
            "root_depth_99": "m",
            # Soil
            "SNDPPT": "percent",
            "CLYPPT": "percent",
            # Topography
            "Length": "m",
            "Area": "km^2",
            "FormFactor": "dimensionless",
            "ShapeFactor": "dimensionless",
            "CompactnessCoefficient": "dimensionless",
            "CirculatoryRatio": "dimensionless",
            "ElongationRatio": "dimensionless",
        }

        # 转换字符串列为分类变量并记录分类映射
        categorical_mappings = {}
        for column in merged_data.columns:
            if merged_data[column].dtype == "object":
                merged_data[column] = merged_data[column].astype("category")
                categorical_mappings[column] = dict(
                    enumerate(merged_data[column].cat.categories)
                )
                merged_data[column] = merged_data[column].cat.codes

        ds = xr.Dataset()
        for column in merged_data.columns:
            attrs = {"units": units_dict.get(column, "unknown")}
            if column in categorical_mappings:
                attrs["category_mapping"] = categorical_mappings[column]

            data_array = xr.DataArray(
                data=merged_data[column].values,
                dims=["basin"],
                # we have set gage_id as index so that it won't be saved as numeric values
                coords={"basin": merged_data.index.values.astype(str)},
                attrs=attrs,
            )
            ds[column] = data_array

        # 将分类映射转换为字符串
        for column in ds.data_vars:
            if "category_mapping" in ds[column].attrs:
                # 将字典转换为字符串
                mapping_str = str(ds[column].attrs["category_mapping"])
                ds[column].attrs["category_mapping"] = mapping_str
        return ds

    def cache_streamflow_xrdataset(self):
        """Save all basins' streamflow data in a netcdf file in the cache directory"""
        variables = self.get_target_cols()
        basins = self.camels_sites["gage_id"].values
        t_range = ["2014-01-01", "2022-01-01"]
        times = [
            hydro_time.t2str(tmp) for tmp in hydro_time.t_range_days(t_range).tolist()
        ]
        data = self.read_target_cols(
            object_ids=basins,
            t_range_list=t_range,
            target_cols=variables,
        )

        return xr.Dataset(
            {
                "streamflow": (
                    ["basin", "time"],
                    data.reshape([len(basins), len(times)])[:, :],
                    {"units": self.streamflow_unit},
                )
            },
            coords={
                "basin": basins,
                "time": pd.to_datetime(times),
            },
        )

    def cache_forcing_xrdataset(self):
        """Save all daymet basin-forcing data in a netcdf file in the cache directory."""
        variables = self.get_relevant_cols()
        basins = self.camels_sites["gage_id"].values
        t_range = ["2014-01-01", "2022-01-01"]
        times = [
            hydro_time.t2str(tmp) for tmp in hydro_time.t_range_days(t_range).tolist()
        ]
        data = self.read_relevant_cols(
            object_ids=basins,
            t_range_list=t_range,
            relevant_cols=variables,
        )
        era5_land_units = {
            "dewpoint_temperature_2m": "K",
            "temperature_2m": "K",
            "skin_temperature": "K",
            "soil_temperature_level_1": "K",
            "soil_temperature_level_2": "K",
            "soil_temperature_level_3": "K",
            "soil_temperature_level_4": "K",
            "lake_bottom_temperature": "K",
            "lake_ice_depth": "m",
            "lake_ice_temperature": "K",
            "lake_mix_layer_depth": "m",
            "lake_shape_factor": "Dimensionless",
            "lake_total_layer_temperature": "K",
            "snow_albedo": "Dimensionless",
            "snow_cover": "%",
            "snow_density": "kg/m^3",
            "snow_depth": "m",
            "snow_depth_water_equivalent": "m",
            "temperature_of_snow_layer": "K",
            "skin_reservoir_content": "m",
            "volumetric_soil_water_layer_1": "Dimensionless",
            "volumetric_soil_water_layer_2": "Dimensionless",
            "volumetric_soil_water_layer_3": "Dimensionless",
            "volumetric_soil_water_layer_4": "Dimensionless",
            "forecast_albedo": "Dimensionless",
            "u_component_of_wind_10m": "m/s",
            "v_component_of_wind_10m": "m/s",
            "surface_pressure": "Pa",
            "leaf_area_index_high_vegetation": "Dimensionless",
            "leaf_area_index_low_vegetation": "Dimensionless",
            "snowfall": "m",
            "snowmelt": "m",
            "surface_latent_heat_flux": "J/m^2",
            "surface_net_solar_radiation": "J/m^2",
            "surface_net_thermal_radiation": "J/m^2",
            "surface_sensible_heat_flux": "J/m^2",
            "surface_solar_radiation_downwards": "J/m^2",
            "surface_thermal_radiation_downwards": "J/m^2",
            # for evaporation and precipitation, we have trans m to mm when reading data
            "evaporation_from_bare_soil": "mm",
            "evaporation_from_open_water_surfaces_excluding_oceans": "mm",
            "evaporation_from_the_top_of_canopy": "mm",
            "evaporation_from_vegetation_transpiration": "mm",
            "potential_evaporation": "mm",
            "runoff": "m",
            "snow_evaporation": "mm",
            "sub_surface_runoff": "m",
            "surface_runoff": "m",
            "total_evaporation": "mm",
            "total_precipitation": "mm",
        }

        return xr.Dataset(
            data_vars={
                **{
                    variables[i]: (
                        ["basin", "time"],
                        data[:, :, i],
                        {"units": era5_land_units[variables[i]]},
                    )
                    for i in range(len(variables))
                }
            },
            coords={
                "basin": basins,
                "time": pd.to_datetime(times),
            },
            attrs={"forcing_type": "era5land"},
        )

    def cache_xrdataset(self):
        """Save all data in a netcdf file in the cache directory"""
        ds_attr = self.cache_attributes_xrdataset()
        ds_attr.to_netcdf(os.path.join(CACHE_DIR, "camelscc_attributes.nc"))
        ds_streamflow = self.cache_streamflow_xrdataset()
        ds_forcing = self.cache_forcing_xrdataset()
        ds = xr.merge([ds_streamflow, ds_forcing])
        ds.to_netcdf(os.path.join(CACHE_DIR, "camelscc_timeseries.nc"))

    def read_ts_xrdataset(
        self,
        gage_id_lst: list = None,
        t_range: list = None,
        var_lst: list = None,
        **kwargs,
    ):
        """read time-series xarray dataset"""
        if var_lst is None:
            return None
        try:
            ts = xr.open_dataset(os.path.join(CACHE_DIR, "camelscc_timeseries.nc"))
        except FileNotFoundError:
            self.cache_xrdataset()
            ts = xr.open_dataset(os.path.join(CACHE_DIR, "camelscc_timeseries.nc"))
        all_vars = ts.data_vars
        if any(var not in ts.variables for var in var_lst):
            raise ValueError(f"var_lst must all be in {all_vars}")
        return ts[var_lst].sel(basin=gage_id_lst, time=slice(t_range[0], t_range[1]))

    def read_attr_xrdataset(self, gage_id_lst=None, var_lst=None, **kwargs):
        """read attribute pandas feather"""
        if var_lst is None or len(var_lst) == 0:
            return None
        try:
            attr = xr.open_dataset(os.path.join(CACHE_DIR, "camelscc_attributes.nc"))
        except FileNotFoundError:
            self.cache_xrdataset()
            attr = xr.open_dataset(os.path.join(CACHE_DIR, "camelscc_attributes.nc"))
        return attr[var_lst].sel(basin=gage_id_lst)

    def read_area(self, gage_id_lst=None):
        """read area of each basin/unit"""
        return self.read_attr_xrdataset(gage_id_lst, ["Area"])

    def read_mean_prcp(self, gage_id_lst=None):
        """read mean precipitation of each basin/unit"""
        return self.read_attr_xrdataset(gage_id_lst, ["p_mean"])


class HydroBasins(HydroData):
    def __init__(self, data_path):
        super().__init__(data_path)

    def get_name(self):
        return " HydroBasins"

    def set_data_source_describe(self):
        self.data_source = "MINIO"
        self.data_source_bucket = "basins-origin"

    def read_BA_xrdataset(self, gage_id_lst: list, var_lst: list, path):
        attr = access_fs.spec_path(path, head="minio")
        if "all" in var_lst:
            return attr.sel(basin=gage_id_lst)
        return attr[var_lst].sel(basin=gage_id_lst)

    def read_MP(self, gage_id_lst, path):
        mean_prep = access_fs.spec_path(path, head="minio")
        return mean_prep["pet_mm_syr"].sel(basin=gage_id_lst)

    def merge_nc_minio_datasets(self, folder_path, basin, var_lst, gap="3h"):
        datasets = []
        basins = []

        file_lst = self.read_file_lst(folder_path)

        for file_path in file_lst:
            basin_id = file_path.split("_")[-1].split(".")[0]
            if basin_id in basin:
                basins.append(basin_id)
                if "ftproot" in file_path:
                    ds = access_fs.spec_path(file_path)
                else:
                    ds = access_fs.spec_path(file_path, head="minio")
                if gap != "1h":
                    ds = self.aggeragate_dataset(ds, gap)
                ds = ds.assign_coords({"basin": basin_id})
                ds = ds.expand_dims("basin")
                datasets.append(ds[var_lst])

        return xr.concat(datasets, dim="basin")

    def aggeragate_dataset(self, ds: xr.Dataset, gap):
        df_res = ds.to_dataframe()
        if "total_evaporation_hourly" in df_res.columns:
            df_res["total_evaporation_hourly"] = (
                df_res["total_evaporation_hourly"].resample(gap, origin="start").sum()
            )
            df_res["total_evaporation_hourly"] *= -1000
            df_res["total_precipitation_hourly"] = (
                df_res["total_precipitation_hourly"].resample(gap, origin="start").sum()
            )
            df_res["total_precipitation_hourly"] *= 1000
        elif "gpm_tp" in df_res.columns:
            df_res["gpm_tp"] = df_res["gpm_tp"].resample(gap, origin="start").sum()
        df_res["streamflow"] = df_res["streamflow"].resample(gap, origin="start").sum()
        df_res = df_res.resample(gap).mean()
        return xr.Dataset.from_dataframe(df_res)

    def read_grid_data(self, file_lst, basin):
        def get_basin_id(file_path):
            return file_path.split("_")[-1].split(".")[0]

        matched_path = next(
            (path for path in file_lst if get_basin_id(path) in basin), None
        )

        if matched_path:
            grid_data = access_fs.spec_path(matched_path, head="minio")
            return grid_data

        return None

    def read_file_lst(self, folder_path):
        if "ftproot" in folder_path:
            return glob.glob(folder_path + "/*")[1:]
        url_path = "s3://" + folder_path
        return conf.FS.glob(url_path + "/**")[1:]
