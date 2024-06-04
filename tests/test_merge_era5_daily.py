import fsspec
import xarray as xr

import hydrodatasource.configs.config as hdscc


def test_merge_era5():
    all_daily_dirs = hdscc.FS.glob('s3://era5-origin/era5/grib/single_levels_tp/**', maxdepth=3)
    daily_dirs = [daily for daily in all_daily_dirs if len(daily.split('/')) == 7]
    for dir in daily_dirs:
        date_nlist = dir.split('/')
        year = date_nlist[-3]
        month = date_nlist[-2]
        day = date_nlist[-1]
        daily_files = hdscc.FS.glob(dir+'/**')[1:]
        daily_ds_list = []
        for dfile in daily_files:
            grib_s3file = f'simplecache::s3://{dfile}'
            time_ds = xr.open_dataset(fsspec.open_local(grib_s3file, s3=hdscc.MINIO_PARAM, filecache=
                {'cache_storage': '/tmp/files'}), engine='cfgrib')
            daily_ds_list.append(time_ds)
        daily_ds = xr.concat(daily_ds_list, 'valid_time')
        hdscc.FS.write_bytes(path=f's3://era5-origin/era5/grib/single_levels_tp_daily/{year}/{month}/{day}.nc',
                             value=daily_ds.to_netcdf())
