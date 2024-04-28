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
        mask = hpm.gen_single_mask(basin, 'gpm')
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
        mask = hpm.gen_single_mask(basin, 'era5_land')
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
        mask = hpm.gen_single_mask(basin, 'smap')
        result_arr = hpm.mean_by_mask(test_smap, 'sm_surface', mask)
        result_arr_list.append(result_arr)
    return result_arr_list


'''
def test_concat_gpm_average():
    basin_id = 'CHN_21401550'
    result_arr_list = test_grid_mean_mask()
    gpm_hour_array = []
    xr_ds = xr.Dataset(coords={'prcpCal_aver': []})
    for i in np.arange(0, len(result_arr_list), 2):
        gpm_hour_i = np.add(result_arr_list[i], result_arr_list[i + 1])
        gpm_hour_array.append(gpm_hour_i)
        temp_ds = xr.Dataset({'prcpCal_aver': gpm_hour_i})
        xr_ds = xr.concat([xr_ds, temp_ds], 'prcpCal_aver')
    tile_path = f's3://basins-origin/hour_data/1h/grid_data/grid_gpm_data/grid_gpm_{basin_id}.nc'
    hdscc.FS.write_bytes(tile_path, xr_ds.to_netcdf())
    return xr_ds


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
    concat_gpm_smap_mean_data(['basin_CHN_songliao_21401550'], [['2020-07-01 00:00:00', '2020-07-31 23:00:00']])


def test_concat_basins_variables():
    basin_ids = ['basin_CHN_songliao_21401550', 'basin_CHN_songliao_21100150', 'basin_CHN_songliao_21110150',
                 'basin_CHN_songliao_21110400', 'basin_CHN_songliao_21113800']
    concat_gpm_smap_mean_data(basin_ids, [['2020-07-01 00:00:00', '2020-07-31 23:00:00']])
