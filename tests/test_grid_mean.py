import time

import fsspec
import geopandas as gpd
import xarray as xr

import hydrodatasource.configs.config as hdscc
import hydrodatasource.processor.mask as hpm
from hydrodatasource.reader.spliter_grid import generate_bbox_from_shp, query_path_from_metadata, \
    concat_gpm_smap_mean_data


def test_grid_mean_mask():
    # 21401550, 碧流河
    test_shp = 's3://basins-origin/basin_shapefiles/basin_CHN_songliao_21401550.zip'
    bbox, basin = generate_bbox_from_shp(test_shp, 'gpm')
    time_start = "2023-06-06 00:00:00"
    time_end = "2023-06-06 02:00:00"
    test_gpm_paths = query_path_from_metadata(time_start, time_end, bbox, data_source="gpm")
    result_arr_list = []
    for path in test_gpm_paths:
        test_gpm = xr.open_dataset(hdscc.FS.open(path))
        mask = hpm.gen_single_mask('basin_CHN_songliao_21401550', basin, 'gpm')
        result_arr = hpm.mean_by_mask(test_gpm, var='precipitationCal', mask=mask)
        result_arr_list.append(result_arr)
    return result_arr_list


def test_grid_mean_era5_land():
    # 21401550, 碧流河
    test_shp = 's3://basins-origin/basin_shapefiles/basin_CHN_songliao_21401550.zip'
    bbox, basin = generate_bbox_from_shp(test_shp, 'era5_land')
    time_start = "2022-06-02"
    time_end = "2022-06-02"
    test_era5_land_paths = query_path_from_metadata(time_start, time_end, bbox, data_source="era5_land")
    result_arr_list = []
    for path in test_era5_land_paths:
        test_era5_land = xr.open_dataset(hdscc.FS.open(path))
        mask = hpm.gen_single_mask('basin_CHN_songliao_21401550', basin, 'era5_land')
        result_arr = hpm.mean_by_mask(test_era5_land, var='tp', mask=mask)
        result_arr_list.append(result_arr)
    return result_arr_list


def test_smap_mean():
    test_shp = 's3://basins-origin/basin_shapefiles/basin_CHN_songliao_21401550.zip'
    bbox, basin = generate_bbox_from_shp(test_shp, 'smap')
    time_start = "2016-02-02"
    time_end = "2016-02-02"
    test_smap_paths = query_path_from_metadata(time_start, time_end, bbox, data_source='smap')
    result_arr_list = []
    for path in test_smap_paths:
        test_smap = xr.open_dataset(hdscc.FS.open(path))
        mask = hpm.gen_single_mask('basin_CHN_songliao_21401550', basin, 'smap')
        result_arr = hpm.mean_by_mask(test_smap, 'sm_surface', mask)
        result_arr_list.append(result_arr)
    return result_arr_list


'''
def test_concat_era5_land_average():
    basin_id = 'CHN_21401550'
    result_arr_list = test_grid_mean_era5_land()
    xr_ds = xr.Dataset(coords={'tp_aver': []})
    # xr_ds是24小时的24个雨量平均值
    for i in np.arange(0, len(result_arr_list)):
        temp_ds = xr.Dataset({'tp_aver': result_arr_list[i]})
        xr_ds = xr.concat([xr_ds, temp_ds], 'tp_aver')
    tile_path = f's3://basins-origin/hour_data/1h/grid_data/grid_era5_land_data/grid_era5_land_{basin_id}.nc'
    hdscc.FS.write_bytes(tile_path, xr_ds.to_netcdf())
    return xr_ds
'''


def test_concat_variables():
    # '3B-HHR-E.MS.MRG.3IMERG.20200701-S000000-E002959.0000.V06B.HDF5_tile.nc4'未区分流域，导致前面的数据被后面的数据覆盖
    basin_ids = ['basin_CHN_songliao_21401550']
    merge_list = concat_gpm_smap_mean_data(basin_ids, [['2022-06-01 00:00:00', '2022-07-31 23:00:00'],
                                                       ['2022-09-01 00:00:00', '2022-09-30 23:00:00']])
    for xr_ds in merge_list:
        hdscc.FS.write_bytes(f's3://basins-origin/hour_data/1h/mean_data/mean_data_merged/mean_data_{basin_ids[0]}',
                             xr_ds.to_netcdf())


def test_concat_basins_variables():
    # 21100150、21110150、21110400、21113800
    # (stations-origin/zz_stations/hour_data/1h/zz_CHN_songliao_{stcd}.csv)数据只有2021.4-2021.7，基本是废站
    basin_ids = ['basin_CHN_songliao_10810201', 'basin_CHN_songliao_20501500', 'basin_CHN_songliao_21113800']
    merge_list = concat_gpm_smap_mean_data(basin_ids, [['2020-07-01 00:00:00', '2020-07-31 23:00:00']])
    for i in range(0, merge_list):
        hdscc.FS.write_bytes(f's3://basins-origin/hour_data/1h/mean_data/mean_data_{basin_ids[i]}',
                             merge_list[i].to_netcdf())


def test_concat_usa_basins_variables():
    basin_usa_ids = ['basin_USA_camels_01491000', 'basin_USA_camels_01548500', 'basin_USA_camels_02014000',
                     'basin_USA_camels_02046000', 'basin_USA_camels_02051500']
    merge_list = concat_gpm_smap_mean_data(basin_usa_ids,
                                           [['2022-07-01 00:00:00', '2022-09-30 23:00:00'],
                                            ['2023-07-01 00:00:00', '2023-09-30 23:00:00']])
    '''
    for i in range(0, merge_list):
        hdscc.FS.write_bytes(f's3://basins-origin/hour_data/1h/mean_data/mean_data_{basin_usa_ids[i]}',
                             merge_list[i].to_netcdf())
    '''
    return merge_list


def test_read_era5_by_basins():
    data_name = 'era5'
    chk_dir = 'single_levels_tp'
    # era5中的经度是0-360°，对西经如美国会出问题
    basin_ids = ['21401550']
    test_chk_dir = f's3://era5-origin/era5/grib/{chk_dir}/2016/02/02/'
    test_paths = hdscc.FS.glob(test_chk_dir + '**')[1:]
    basin_tgdf = gpd.read_file(hdscc.FS.open('s3://basins-origin/basins_shp.zip'))
    basin_gdfs = basin_tgdf[basin_tgdf['BASIN_ID'].isin(basin_ids)]
    basin_mean_dict = {}
    time_mean_list = []
    start_cyc_time = time.time()
    for basin_id in basin_ids:
        for time_path in test_paths:
            basin_part = basin_gdfs[basin_gdfs['BASIN_ID'] == basin_id]
            mask = hpm.gen_single_mask(basin_id, basin_part, data_name)
            bbox = [mask['lon'].values.min(), mask['lon'].values.max(), mask['lat'].values.max(),
                    mask['lat'].values.min()]
            time_path = f'simplecache::s3://{time_path}'
            time_ds = xr.open_dataset(fsspec.open_local(time_path, s3=hdscc.MINIO_PARAM, filecache=
                {'cache_storage': '/tmp/files'}), engine='cfgrib')
            time_ds_split = time_ds.sel(longitude=slice(bbox[0], bbox[1]), latitude=slice(bbox[2], bbox[3]))
            mean_res = hpm.mean_by_mask(src=time_ds_split, var='tp', mask=mask)
            time_mean_list.append(mean_res)
        basin_mean_dict[basin_id] = time_mean_list
    use_time = time.time() - start_cyc_time
    print(use_time)
    return basin_mean_dict

'''
def split_dataset(time_path, bboxes: list):
    time_path = f'simplecache::s3://{time_path}'
    time_ds = xr.open_dataset(fsspec.open_local(time_path, s3=hdscc.MINIO_PARAM, filecache=
    {'cache_storage': '/tmp/files'}), engine='cfgrib')
    datasets = [time_ds.sel(longitude=slice(bbox[0], bbox[1]), latitude=slice(bbox[2], bbox[3])) for bbox in bboxes]
    result = xr.concat(datasets, dim='time')
    return result
'''
