import pandas as pd
import xarray as xr

import hydrodatasource.configs.config as conf


def query_path_from_metadata(time_start=None, time_end=None, bbox=None):
    # query path from other columns from metadata.csv
    metadata_df = pd.read_csv('metadata.csv')
    paths = metadata_df
    if time_start is not None:
        paths = paths[paths['time_start'] >= time_start]
    if time_end is not None:
        paths = paths[paths['time_end'] <= time_end]
    if bbox is not None:
        paths = paths[
            (paths['bbox'].apply(lambda x: string_to_list(x)[0] <= bbox[0])) &
            (paths['bbox'].apply(lambda x: string_to_list(x)[1] >= bbox[1])) &
            (paths['bbox'].apply(lambda x: string_to_list(x)[2] >= bbox[2])) &
            (paths['bbox'].apply(lambda x: string_to_list(x)[3] <= bbox[3]))]
    for path in paths['path']:
        path_ds = xr.open_dataset(conf.FS.open(path))
        tile_ds = path_ds.sel(time=slice(time_start, time_end), lon=slice(bbox[0], bbox[1]),
                              lat=slice(bbox[3], bbox[2]))
        # 会覆盖源数据，注意
        tile_ds.to_netcdf(path.rstrip('.nc4') + '_tile.nc4')
    return paths


def test_query_path_from_metadata():
    time_start = '2018-06-05 01:00:00'
    time_end = '2018-06-05 02:00:00'
    bbox = [-110, -69, 47, 26]
    paths = query_path_from_metadata(time_start, time_end, bbox)
    return paths


def string_to_list(x: str):
    return list(map(float, x[1:-1].split(',')))

