import fsspec
import xarray as xr
import glob
import hydrodatasource.configs.config as hdscc
import numpy as np
import geopandas as gpd
from hydrodatasource.configs.config import FS


def test_merge_era5():
    all_daily_dirs = hdscc.FS.glob('s3://era5-origin/era5/grib/single_levels_tp/**', maxdepth=3)
    daily_dirs = [daily for daily in all_daily_dirs if len(daily.split('/')) == 7]
    for dir in daily_dirs:
        date_nlist = dir.split('/')
        year = date_nlist[-3]
        month = date_nlist[-2]
        day = date_nlist[-1]
        daily_files = hdscc.FS.glob(dir + '/**')[1:]
        daily_ds_list = []
        for dfile in daily_files:
            grib_s3file = f'simplecache::s3://{dfile}'
            time_ds = xr.open_dataset(fsspec.open_local(grib_s3file, s3=hdscc.MINIO_PARAM, filecache=
            {'cache_storage': '/tmp/files'}), engine='cfgrib')
            daily_ds_list.append(time_ds)
        daily_ds = xr.concat(daily_ds_list, 'valid_time')
        hdscc.FS.write_bytes(path=f's3://era5-origin/era5/grib/single_levels_tp_daily/{year}/{month}/{day}.nc',
                             value=daily_ds.to_netcdf())


def test_merge_era5l_camels_sl_sd():
    import pandas as pd
    mistake_list = ['10901810', '11405810', '20101940', '20302520']
    old_era5l_ds_15_23 = xr.open_dataset("/ftproot/era5land/563_basins_era5_land_2015_2023.nc")
    old_era5l_ds_15_23 = old_era5l_ds_15_23.astype(np.float32)
    old_era5l_ds_15_23['time_start'] = pd.to_datetime(old_era5l_ds_15_23['time_start'])
    old_era5l_ds_15_23 = old_era5l_ds_15_23.drop_sel(basin_id=mistake_list)
    old_era5l_ds_00_14 = xr.open_dataset("/ftproot/camels_songliao_era5land_2000_2014.nc")
    old_era5l_ds_00_14 = old_era5l_ds_00_14.drop_sel(basin_id=mistake_list).drop('Unnamed: 0')
    old_era5l_ds = xr.concat([old_era5l_ds_00_14, old_era5l_ds_15_23], dim='time_start')
    old_era5l_ds = old_era5l_ds.astype(np.float32)
    sd_ds = xr.open_dataset("/ftproot/songliao_shandong_basins_era5land.nc")
    new_era5l_ds = xr.concat([old_era5l_ds, sd_ds], dim='basin_id')
    new_era5l_ds = new_era5l_ds.astype(np.float32)
    # 不要从自己的工作文件夹中转，直接生成到相应文件夹即可，以防cp命令改变数据
    new_era5l_ds.to_netcdf('/ftproot/632_basins_era5land_fixed.nc')


def test_merge_new_streamflow():
    import pandas as pd
    new_dfs = glob.glob('/ftproot/basins-origin/basins-streamflow-BSAD/basins-streamflow-with BSAD/*.csv',
                        recursive=True)
    basins_shp = gpd.read_file(hdscc.FS.open('s3://basins-origin/basins_shp.zip'))
    rsvr_dfs = []
    for rsvr_file in new_dfs:
        rsvr_part_df = pd.read_csv(rsvr_file, engine='c', parse_dates=['TM'])
        if 'rsvr' in rsvr_file:
            rsvr_part_df['STCD'] = rsvr_part_df['STCD'].astype(str)
            rsvr_part_df = rsvr_part_df.rename(columns={'INQ': 'Q'})
            rsvr_part_df = rsvr_part_df[['STCD', 'TM', 'Q']]
            stcd_area = get_area_from_stcd(basins_shp, rsvr_part_df['STCD'].to_list()[0]).to_list()[0]
            rsvr_part_df['Q'] = rsvr_part_df['Q'].apply(lambda x: x / stcd_area * 3.6)
            rsvr_dfs.append(rsvr_part_df)
        else:
            rsvr_part_df = rsvr_part_df.rename(columns={'INQ': 'Q'})
            rsvr_part_df['STCD'] = np.repeat(rsvr_file.split('/')[-1].split('.')[0], len(rsvr_part_df))
            stcd_area = get_area_from_stcd(basins_shp, rsvr_part_df['STCD'].to_list()[0]).to_list()[0]
            rsvr_part_df['Q'] = rsvr_part_df['Q'].apply(lambda x: x / stcd_area * 3.6)
            rsvr_part_df = rsvr_part_df[['STCD', 'TM', 'Q']]
            rsvr_dfs.append(rsvr_part_df)
    rsvr_df = pd.concat(rsvr_dfs).rename(columns={'TM': 'time_start', 'STCD': 'basin_id', 'Q': 'streamflow'})
    rsvr_df = rsvr_df[(rsvr_df['time_start'] >= pd.to_datetime('2000-01-01 00:00:00')) & (
            rsvr_df['time_start'] <= pd.to_datetime('2023-12-31 23:00:00'))]
    rsvr_df['basin_id'] = rsvr_df['basin_id'].astype(str)
    rsvr_df = rsvr_df.set_index(['time_start', 'basin_id'])
    # 在这里xarray可能会莫名生成几个分钟数据，造成索引错误
    rsvr_ds = xr.Dataset.from_dataframe(rsvr_df)
    time_start_arr = rsvr_ds['time_start'].to_numpy()
    hourly_dr = pd.date_range(time_start_arr[0], time_start_arr[-1], freq='1h')
    rsvr_ds = rsvr_ds.sel(time_start=hourly_dr)
    # rsvr_ds.coords['basin_id'] = rsvr_df['basin_id'].to_numpy()
    rsvr_ds.to_netcdf("/ftproot/100_basins_streamflow.nc")


def get_area_from_stcd(basins_shp, stcd: str):
    geom_stcd = basins_shp.geometry[basins_shp['BASIN_ID'] == stcd]
    return geom_stcd.area * 12345.6789


def test_compare_basin_amount():
    total_shps = gpd.read_file(FS.open('s3://basins-origin/basins_shp.zip'))
    total_basin_ids = total_shps['BASIN_ID'].to_numpy()
    usa_stream_basins = xr.open_dataset("/ftproot/516_basins_2000_2014.nc")['basin_id'].to_numpy()
    chn_stream_basins = xr.open_dataset("/ftproot/100_basins_streamflow.nc")['basin_id'].to_numpy()
    calced_basins = np.concatenate([usa_stream_basins, chn_stream_basins])
    result_diff = np.setdiff1d(total_basin_ids, calced_basins)
    print(result_diff)
    return result_diff


def test_merge_616_basins_streamflow():
    usa_basins_53_15_23 = xr.open_dataset("/ftproot/53_usa_basins_streamflow.nc")
    usa_basins_463_15_23 = xr.open_dataset("/ftproot/usgs_datas_462_basins_after_2019/463_basins_2015_2023.nc")
    usa_basins_516_00_14 = xr.open_dataset("/ftproot/516_basins_2000_2014.nc")
    chn_basins_100_00_23 = xr.open_dataset("/ftproot/100_basins_streamflow.nc")
    chn_basins_100_00_23['time_start'] = chn_basins_100_00_23['time_start'] - np.timedelta64(8, 'h')
    usa_basins_516_15_23 = xr.concat([usa_basins_463_15_23, usa_basins_53_15_23], dim='basin_id')
    usa_basins = xr.concat([usa_basins_516_00_14, usa_basins_516_15_23], dim='time_start')
    basins_ds = xr.concat([usa_basins.drop_duplicates('time_start'), chn_basins_100_00_23], dim='basin_id')
    basins_ds = basins_ds.astype(np.float32)
    basins_ds.coords['basin_id'] = basins_ds['basin_id'].to_numpy()
    basins_ds.to_netcdf('/ftproot/616_basins_streamflow_2000_2023.nc')
