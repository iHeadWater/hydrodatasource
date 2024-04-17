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
