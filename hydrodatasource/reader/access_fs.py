import glob
import logging
import os

import fsspec
import geopandas as gpd
import intake as intk
import pandas as pd
import ujson
import xarray as xr
import zarr
from kerchunk.hdf import SingleHdf5ToZarr
import hydrodatasource.configs.config as conf


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
    obj (str): The file path or URL of the data, format is 's3://bucket_name/directory_name/file_name' in minio.
    storage_option (dict, optional): The storage options for accessing the data. Defaults to None, if you want to
                                     read from minio, storage_option should be minio login params.
                                     See hydrodatasource.configs.config.MINIO_PARAM to get reference
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
            data_obj = xr.Dataset(data_obj)
            if (need_cache is True) & (storage_option is not None):
                data_obj.to_csv(path_or_buf=os.path.join(conf.LOCAL_DATA_PATH, cache_name))
        elif (ext_name in ['nc', 'nc4', 'h5', 'hdf5']) or ('nc4' in obj):
            if need_refer is True:
                data_obj = gen_refer_and_read_zarr(obj, storage_option=storage_option)
            else:
                '''
                nc_source = intk.datatypes.HDF5(obj, storage_options=storage_option)
                nc_src_reader = intk.readers.DaskHDF(nc_source).to_reader()
                data_obj: xr.Dataset = nc_src_reader.read()
                '''
                if storage_option is not None:
                    if (ext_name == 'nc') or (ext_name == 'nc4') or ('nc4' in obj):
                        data_obj = xr.open_dataset(conf.FS.open(obj), chunks='auto')
                    elif (ext_name == 'hdf5') or (ext_name == 'h5'):
                        data_obj = xr.open_dataset(conf.FS.open(obj), engine='h5netcdf', chunks='auto', phony_dims='access')
                else:
                    if (ext_name == 'nc') or (ext_name == 'nc4') or ('nc4' in obj):
                        data_obj = xr.open_dataset(obj, chunks='auto')
                    elif (ext_name == 'hdf5') or (ext_name == 'h5'):
                        data_obj = xr.open_dataset(obj, engine='h5netcdf', chunks='auto', phony_dims='access')
            if (need_cache is True) & (storage_option is not None):
                data_obj.to_netcdf(path=os.path.join(conf.LOCAL_DATA_PATH, cache_name))
        elif ext_name == 'json':
            data_obj = pd.read_json(obj, storage_options=storage_option)
            if (need_cache is True) & (storage_option is not None):
                data_obj.to_json(path_or_buf=os.path.join(conf.LOCAL_DATA_PATH, cache_name))
        elif ext_name == 'zip':
            # Now zipfile is used to read shapefile
            # Can't read shapefile directly, see this: https://github.com/geopandas/geopandas/issues/3129
            # pip install pyogrio
            if storage_option is not None:
                data_obj = gpd.read_file(conf.FS.open(obj), engine='pyogrio')
            else:
                data_obj = gpd.read_file(obj)
            if (need_cache is True) & (storage_option is not None):
                data_obj.to_file(path=os.path.join(conf.LOCAL_DATA_PATH, cache_name))
        elif 'grb2' in obj:
            if storage_option is not None:
                obj = f'simplecache::{obj}'
                grib_ds = xr.open_dataset(fsspec.open_local(obj, s3=conf.MINIO_PARAM, filecache=
                        {'cache_storage': '/tmp/files'}), engine='cfgrib')
            else:
                grib_ds = xr.open_dataset(obj)
            if (need_cache is True) & (storage_option is not None):
                grib_ds.to_netcdf(path=os.path.join(conf.LOCAL_DATA_PATH, cache_name))
        elif ext_name == 'txt':
            data_obj = pd.read_fwf(obj, storage_options=storage_option)
            if (need_cache is True) & (storage_option is not None):
                data_obj.to_csv(os.path.join(conf.LOCAL_DATA_PATH, cache_name))
        elif ext_name == 'zarr':
            if storage_option is not None:
                zarr_mapper = conf.FS.get_mapper(obj)
                # KVStore is introduced in zarr specification V3
                # https://zarr-specs.readthedocs.io/en/latest/v3/stores.html
                zarr_store = zarr.storage.KVStore(zarr_mapper)
                data_obj = xr.open_zarr(zarr_store)
            else:
                data_obj = xr.open_zarr(obj)
        else:
            logging.error(f'Unsupported file type: {ext_name}')
    else:
        data_obj = object()
        logging.error("这是数据存储，不是百度云盘！")
    return data_obj


def gen_refer_and_read_zarr(obj_path, storage_option=None):
    # https://github.com/fsspec/kerchunk/discussions/431
    obj_json_path = 's3://references/' + str(obj_path) + '.json'
    if not conf.FS.exists(obj_json_path):
        obj_path = 's3://' + obj_path
        with conf.FS.open(obj_path, 'rb') as fpj:
            nc_chunks = SingleHdf5ToZarr(fpj, obj_path)
            with conf.FS.open(obj_json_path, 'wb') as fp:
                fp.write(ujson.dumps(nc_chunks.translate()).encode())
    data_type_obj = intk.datatypes.HDF5(obj_json_path, storage_options=storage_option)
    data_reader_obj = intk.readers.XArrayDatasetReader(data_type_obj).to_reader()
    data_obj = data_reader_obj.read()
    return data_obj
