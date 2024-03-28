import hydrodatasource.configs.config as conf
import xarray as xr
import numpy as np
import pandas as pd


def test_read_gpm_csv():
    gpm_nc_files = [file for file in conf.FS.glob('s3://grids-origin/GPM/**') if 'nc4' in file]
    bbox_list = []
    start_list = []
    end_list = []
    res_lon_list = []
    res_lat_list = []
    for i in range(0, len(gpm_nc_files)):
        test_ds_i = xr.open_dataset(conf.FS.open(gpm_nc_files[i]))
        bbox = [np.min(test_ds_i['lon'].to_numpy()), np.max(test_ds_i['lon'].to_numpy()),
                np.max(test_ds_i['lat'].to_numpy()), np.min(test_ds_i['lat'].to_numpy())]
        bbox_list.append(bbox)
        start_list.append(test_ds_i['time'].to_numpy()[0])
        end_list.append(test_ds_i['time'].to_numpy()[-1])
        lon_res = abs(np.diff(test_ds_i['lon'].to_numpy())[0])
        res_lon_list.append(lon_res)
        lat_res = abs(np.diff(test_ds_i['lat'].to_numpy())[0])
        res_lat_list.append(lat_res)
    test_pd = pd.DataFrame({'bbox': bbox_list, 'time_start': start_list, 'time_end': end_list,
                            'res_lon': lon_res, 'res_lat': lat_res, 'path': ['s3://' + file for file in gpm_nc_files]})
    test_pd.to_csv('gpm_metadata.csv')
