import json
import logging
import os

import geopandas as gpd
import intake as itk
import pandas as pd
import xarray as xr

import hydrodata.configs.config as conf


def spec_path(url_path: str, head='local', need_cache=False):
    if head == 'local':
        url_path = os.path.join(conf.LOCAL_DATA_PATH, url_path)
        ret_data = read_valid_data(url_path, need_cache=need_cache)
    elif head == 'minio':
        url_path = 's3://' + url_path
        ret_data = read_valid_data(url_path, storage_option=conf.MINIO_PARAM, need_cache=need_cache)
    else:
        raise ValueError("head should be 'local' or 'minio'")
    return ret_data


# Extract from HydroForecast
def read_valid_data(obj: str, storage_option=None, need_cache=False):
    """
    Read valid data from different file types.
    See https://intake.readthedocs.io/en/latest/plugin-directory.html
    pip install intake-xarray intake-geopandas

    Parameters:
    obj (str): The file path or URL of the data.
    storage_option (dict, optional): The storage options for accessing the data. Defaults to None.
    need_cache (bool, optional): Whether to cache the data. Defaults to False.

    Returns:
    object: The data object.
    """
    data_obj = None
    dot_in_obj = '.' in obj
    cache_name = obj.lstrip('s3://').split('/')[-1]
    if not dot_in_obj:
        txt_source = itk.open_textfiles(obj, storage_options=storage_option)
        data_obj = txt_source.read()
        if (need_cache is True) & (storage_option is not None):
            data_obj.to_file(path=os.path.join(conf.LOCAL_DATA_PATH, cache_name))
    elif dot_in_obj:
        ext_name = obj.split('.')[-1]
        if ext_name == 'csv':
            data_obj = pd.read_csv(obj, storage_options=storage_option)
            if (need_cache is True) & (storage_option is not None):
                data_obj.to_csv(path=os.path.join(conf.LOCAL_DATA_PATH, cache_name))
        elif ext_name == 'nc' or ext_name == 'nc4':
            nc_source = itk.open_netcdf(obj, storage_options=storage_option)
            data_obj: xr.Dataset = nc_source.read()
            if (need_cache is True) & (storage_option is not None):
                data_obj.to_netcdf(path=os.path.join(conf.LOCAL_DATA_PATH, cache_name))
        elif ext_name == 'json':
            # json_source = itk.open_json(obj, storage_options=storage_option)
            data_obj = pd.read_json(obj, storage_options=storage_option)
            # data_obj = json_source.read()
            if (need_cache is True) & (storage_option is not None):
                data_obj.to_json(path_or_buf=os.path.join(conf.LOCAL_DATA_PATH, cache_name))
        elif ext_name == 'shp':
            # Can't run directly, see this: https://github.com/geopandas/geopandas/issues/3129
            remote_shp_obj = conf.FS.open(obj)
            data_obj = gpd.read_file(remote_shp_obj, engine='pyogrio')
            if (need_cache is True) & (storage_option is not None):
                data_obj.to_file(path=os.path.join(conf.LOCAL_DATA_PATH, cache_name))
        elif 'grb2' in obj:
            # ValueError: unrecognized engine cfgrib must be one of: ['netcdf4', 'h5netcdf', 'scipy', 'store', 'zarr']
            # https://blog.csdn.net/weixin_44052055/article/details/108658464?spm=1001.2014.3001.5501
            # 似乎只能用conda来装eccodes
            remote_grib_obj = conf.FS.open(obj)
            grib_ds = xr.open_dataset(remote_grib_obj)
            if (need_cache is True) & (storage_option is not None):
                grib_ds.to_netcdf(path=os.path.join(conf.LOCAL_DATA_PATH, cache_name))
        elif ext_name == 'txt':
            # txt_source = itk.open_textfiles(obj, storage_options=storage_option)
            data_obj = pd.read_fwf(obj, storage_options=storage_option)
            # data_obj = txt_source.read()
            if (need_cache is True) & (storage_option is not None):
                data_obj.to_csv(os.path.join(conf.LOCAL_DATA_PATH, cache_name))
        else:
            logging.error(f'Unsupported file type: {ext_name}')
    else:
        data_obj = object()
        logging.error("这是数据存储，不是百度云盘！")
    return data_obj
