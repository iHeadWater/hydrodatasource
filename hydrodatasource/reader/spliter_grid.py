import os
import numpy as np
import pandas as pd
import xarray as xr
from pandas.core.indexes.api import default_index
from hydrodatasource.processor.mask import gen_single_mask
import hydrodatasource.configs.config as conf
from hydrodatasource.reader import access_fs

def query_path_from_metadata(time_start=None, time_end=None, bbox=None, data_source='gpm'):
    # query path from other columns from metadata.csv
    metadata_df = pd.read_csv('metadata.csv')
    tile_list = []
    paths = metadata_df
    if time_start is not None:
        paths = paths[paths['time_start'] >= time_start]
    if time_end is not None:
        paths = paths[paths['time_end'] <= time_end]
    if data_source == 'gpm':
        if bbox is not None:
            paths = paths[
                (paths['bbox'].apply(lambda x: string_to_list(x)[0] <= bbox[0])) &
                (paths['bbox'].apply(lambda x: string_to_list(x)[1] >= bbox[1])) &
                (paths['bbox'].apply(lambda x: string_to_list(x)[2] >= bbox[2])) &
                (paths['bbox'].apply(lambda x: string_to_list(x)[3] <= bbox[3]))]
    elif data_source == 'gfs':
        path_list_predicate = paths[paths['path'].isin(choose_gfs(paths, time_start, time_end))]
        paths = paths[path_list_predicate['bbox'].apply(lambda x: string_to_list(x)[0] <= bbox[0]) &
                      (paths['bbox'].apply(lambda x: string_to_list(x)[1] >= bbox[1])) &
                      (paths['bbox'].apply(lambda x: string_to_list(x)[2] >= bbox[2])) &
                      (paths['bbox'].apply(lambda x: string_to_list(x)[3] <= bbox[3]))]
    for path in paths['path']:
        path_ds = xr.open_dataset(conf.FS.open(path))
        if data_source == 'gpm':
            tile_ds = path_ds.sel(time=slice(time_start, time_end), lon=slice(bbox[0], bbox[1]),
                                  lat=slice(bbox[3], bbox[2]))
        # 会扰乱桶，注意
        elif data_source == 'gfs':
            tile_ds = path_ds.sel(time=slice(time_start, time_end), longitude=slice(bbox[0], bbox[1]),
                                  latitude=slice(bbox[2], bbox[3]))
        else:
            tile_ds = path_ds
        tile_path = path.rstrip('.nc4') + '_tile.nc4'
        tile_list.append(tile_path)
        temp_df = pd.DataFrame(
            {'bbox': str(bbox), 'time_start': time_start, 'time_end': time_end, 'res_lon': 0.25, 'res_lat': 0.25,
             'path': tile_path}, index=default_index(1))
        metadata_df = pd.concat([metadata_df, temp_df], axis=0)
        if data_source == 'gpm':
            conf.FS.write_bytes(tile_path, tile_ds.to_netcdf())
        elif data_source == 'gfs':
            tile_ds.to_netcdf('temp.nc4')
            conf.FS.put_file('temp.nc4', tile_path)
            os.remove('temp.nc4')
    metadata_df.to_csv('metadata.csv', index=False)
    return tile_list


def choose_gfs(paths, start_time, end_time):
    """
        This function chooses GFS data within a specified time range and bounding box.
        Args:
            start_time (datetime, YY-mm-dd): The start time of the desired data.
            end_time (datetime, YY-mm-dd): The end time of the desired data.
            bbox (list): A list of four coordinates representing the bounding box.
        Returns:
            list: A list of GFS data within the specified time range and bounding box.
        """
    path_list = []
    produce_times = ['00', '06', '12', '18']
    if start_time is None:
        start_time = paths['time_start'].iloc[0]
    if end_time is None:
        end_time = paths['time_end'].iloc[-1]
    time_range = pd.date_range(start_time, end_time, freq='1D')
    for date in time_range:
        date_str = date.strftime('%Y/%m/%d')
        for i in range(len(produce_times)):
            for j in range(6 * i, 6 * (i + 1)):
                path = 's3://grids-origin/GFS/GEE/1h/' + date_str + '/' + produce_times[i] + '/gfs20220103.t' + \
                       produce_times[i] + 'z.nc4.0p25.f' + '{:03d}'.format(j)
                path_list.append(path)
    return path_list


def string_to_list(x: str):
    return list(map(float, x[1:-1].split(',')))


def generate_bbox_from_shp(basin_shape_path):
    basin_id = basin_shape_path.split('/')[-1].split('.')[0]
    mask = gen_single_mask(basin_id=basin_id, shp_path=basin_shape_path, dataname='gfs', mask_path='temp_mask', minio=True)
    bbox = [mask['lon'].values.min(), mask['lon'].values.max(), mask['lat'].values.max(), mask['lat'].values.min()]
    return mask, bbox

def merge_with_spatial_average(gpm_file, gfs_file, smap_file, output_file_path):
    def calculate_and_rename(input_file_path, prefix):
        ds = access_fs.spec_path(input_file_path, head="minio")
        avg_ds = ds.mean(dim=["lat", "lon"], skipna=True).astype("float32")
        new_names = {var_name: (f"{prefix}_tp" if var_name in ["tp", "__xarray_dataarray_variable__"] else f"{prefix}_{var_name}") for var_name in avg_ds.data_vars}
        avg_ds_renamed = avg_ds.rename(new_names)
        return avg_ds_renamed

    basin_id = output_file_path.split('_')[-1].split('.')[0]
    
    gfs_avg_renamed = calculate_and_rename(gfs_file, "gfs")
    gpm_avg_renamed = calculate_and_rename(gpm_file, "gpm")
    smap_avg_renamed = calculate_and_rename(smap_file, "smap")

    intersect_time = np.intersect1d(gfs_avg_renamed.time.values, gpm_avg_renamed.time.values, assume_unique=True)
    intersect_time = np.intersect1d(intersect_time, smap_avg_renamed.time.values, assume_unique=True)

    gfs_intersected = gfs_avg_renamed.sel(time=intersect_time)
    gpm_intersected = gpm_avg_renamed.sel(time=intersect_time)
    smap_intersected = smap_avg_renamed.sel(time=intersect_time)

    merged_ds = xr.merge([gfs_intersected, gpm_intersected, smap_intersected])
    merged_ds = merged_ds.assign_coords({"basin": basin_id}).expand_dims("basin")

    conf.FS.write_bytes(output_file_path, merged_ds.to_netcdff())
    return merged_ds
