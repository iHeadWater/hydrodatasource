import numpy as np

import hydrodatasource.processor.mask as hpm
from hydrodatasource.reader.spliter_grid import generate_bbox_from_shp, query_path_from_metadata
import hydrodatasource.configs.config as hdscc
import xarray as xr


def test_grid_mean_mask():
    # 21401550, 碧流河
    test_shp = 's3://basins-origin/basin_shapefiles/basin_CHN_songliao_21401550.zip'
    mask, bbox = generate_bbox_from_shp(test_shp, 'gpm')
    time_start = "2023-06-06 00:00:00"
    time_end = "2023-06-06 02:00:00"
    test_gpm_paths = query_path_from_metadata(time_start, time_end, bbox, data_source="gpm")
    result_arr_list = []
    for path in test_gpm_paths:
        test_gpm = xr.open_dataset(hdscc.FS.open(path))
        result_arr = hpm.mean_by_mask(test_gpm, var='precipitationCal', mask=mask)
        result_arr_list.append(result_arr)
    return result_arr_list


def test_concat_gpm_average():
    basin_id = 'CHN_21401550'
    result_arr_list = test_grid_mean_mask()
    gpm_hour_array = []
    xr_ds = xr.Dataset(coords={'prcpCal_aver':[]})
    for i in np.arange(0, len(result_arr_list), 2):
        gpm_hour_i = np.add(result_arr_list[i], result_arr_list[i+1])
        gpm_hour_array.append(gpm_hour_i)
        temp_ds = xr.Dataset({'prcpCal_aver': gpm_hour_i})
        xr_ds = xr.concat([xr_ds, temp_ds], 'prcpCal_aver')
    tile_path = f's3://basins-origin/hour_data/1h/grid_data/grid_gpm_data/grid_gpm_{basin_id}.nc'
    hdscc.FS.write_bytes(tile_path, xr_ds.to_netcdf())
    return xr_ds
