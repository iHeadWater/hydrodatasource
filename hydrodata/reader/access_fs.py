import glob
import logging
import os

import geopandas as gpd
import intake as intk
import pandas as pd
import ujson
import xarray as xr
import zarr
from kerchunk.hdf import SingleHdf5ToZarr

import hydrodata.configs.config as conf


def spec_path(url_path: str, head='local', need_cache=False, is_dir=False):
    if is_dir is False:
        if head == 'local':
            url_path = os.path.join(conf.LOCAL_DATA_PATH, url_path)
            ret_data = read_valid_data(url_path, need_cache=need_cache)
        elif head == 'minio':
            url_path = 's3://' + url_path
            ret_data = read_valid_data(url_path, storage_option=conf.MINIO_PARAM, need_cache=need_cache)
        else:
            raise ValueError("head should be 'local' or 'minio'")
    else:
        ret_data = []
        if head == 'local':
            url_path = os.path.join(conf.LOCAL_DATA_PATH, url_path)
            for file in glob.glob(url_path + '/**', recursive=True):
                if not os.path.isdir(file):
                    ret_data.append(read_valid_data(file, need_cache=need_cache))
        elif head == 'minio':
            url_path = 's3://' + url_path
            for file in conf.FS.glob(url_path + '/**'):
                if not conf.FS.isdir(file):
                    ret_data.append(read_valid_data(file, storage_option=conf.MINIO_PARAM, need_cache=need_cache))
        else:
            raise ValueError("head should be 'local' or 'minio'")
    return ret_data


# Extract from HydroForecast
def read_valid_data(obj: str, storage_option=None, need_cache=False, need_refer=False):
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
        data_obj = pd.read_fwf(obj, storage_options=storage_option)
        if (need_cache is True) & (storage_option is not None):
            data_obj.to_file(path=os.path.join(conf.LOCAL_DATA_PATH, cache_name))
    elif dot_in_obj:
        ext_name = obj.split('.')[-1]
        if ext_name == 'csv':
            data_obj = pd.read_csv(obj, storage_options=storage_option)
            if (need_cache is True) & (storage_option is not None):
                data_obj.to_csv(path=os.path.join(conf.LOCAL_DATA_PATH, cache_name))
        elif (ext_name == 'nc') or (ext_name == 'nc4') or (ext_name == 'hdf5'):
            if need_refer is True:
                data_obj = gen_refer_and_read_zarr(obj, storage_option=storage_option)
            else:
                nc_source = intk.datatypes.HDF5(obj, storage_options=storage_option)
                nc_src_reader = intk.readers.DaskHDF(nc_source).to_reader()
                data_obj: xr.Dataset = nc_src_reader.read()
            if (need_cache is True) & (storage_option is not None):
                data_obj.to_netcdf(path=os.path.join(conf.LOCAL_DATA_PATH, cache_name))
        elif ext_name == 'json':
            data_obj = pd.read_json(obj, storage_options=storage_option)
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
            data_obj = pd.read_fwf(obj, storage_options=storage_option)
            if (need_cache is True) & (storage_option is not None):
                data_obj.to_csv(os.path.join(conf.LOCAL_DATA_PATH, cache_name))
        elif ext_name == '.zarr':
            zarr_mapper = conf.FS.get_mapper(obj)
            zarr_store = zarr.storage.KVStore(zarr_mapper)
            data_obj = xr.open_zarr(zarr_store)
        else:
            logging.error(f'Unsupported file type: {ext_name}')
    else:
        data_obj = object()
        logging.error("这是数据存储，不是百度云盘！")
    return data_obj


def gen_refer_and_read_zarr(obj_path, storage_option=None):
    # 有问题尚未解决
    obj_json_path = 's3://references/' + str(obj_path) + '.json'
    if not conf.FS.exists(obj_json_path):
        with open(obj_path, 'wb') as fpj:
            nc_chunks = SingleHdf5ToZarr(h5f=obj_path)
            fpj.write(ujson.dumps(nc_chunks.translate()).encode())
    data_type_obj = intk.datatypes.HDF5(obj_json_path, storage_options=storage_option)
    data_reader_obj = intk.readers.XArrayDatasetReader(data_type_obj).to_reader()
    data_obj = data_reader_obj.read()
    return data_obj


def filter_area_from_minio(aoi_type, aoi_param, data_name, state='origin', need_cache=False):
    # todo
    minio_path = aoi_type + '/' + aoi_param + '/' + data_name
