import os

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
import xarray as xr
import tzfpy
import dataretrieval.nwis as nwis

from hydrodatasource.reader.spliter_grid import read_streamflow_from_minio
from hydrodatasource.configs.config import FS


def test_gen_camels_hourly_shp():
    camels_shp = '/ftproot/camels/camels_us/basin_set_full_res/HCDN_nhru_final_671.shp'
    camels_hourly_csvs = '/ftproot/camels_hourly/data/usgs_streamflow_csv/'
    csv_paths = os.listdir(camels_hourly_csvs)
    minio_csv_paths = FS.glob('s3://basins-origin/basin_shapefiles/**')[1:]
    minio_stcds = [csv.split('_')[-1].split('.')[0] for csv in minio_csv_paths]
    hourly_basin_ids = [path.split('-')[0] for path in csv_paths]
    camels_gpd = gpd.read_file(camels_shp)
    camels_gpd['hru_id'] = camels_gpd['hru_id'].astype('str')
    camels_gpd['hru_id'] = camels_gpd['hru_id'].apply(lambda x: x.zfill(8))
    camels_hourly_gpd_blank = camels_gpd[camels_gpd['hru_id'].isin(hourly_basin_ids)]
    camels_hourly_gpd = camels_hourly_gpd_blank[~camels_gpd['hru_id'].isin(minio_stcds)]
    camels_hourly_gpd.to_file('/home/jiaxuwu/camels_hourly_basins.shp')


def test_read_usa_streamflow():
    # basin_list = hdscc.FS.glob('s3://basins-origin/basin_shapefiles/**')
    basin_list = ["basin_USA_camels_01181000", "basin_USA_camels_01411300",
                  "basin_USA_camels_01414500", "basin_USA_camels_02016000",
                  "basin_USA_camels_02018000", "basin_USA_camels_02481510",
                  "basin_USA_camels_03070500", "basin_USA_camels_08324000",
                  "basin_USA_camels_11266500", "basin_USA_camels_11523200",
                  "basin_USA_camels_12020000", "basin_USA_camels_12167000",
                  "basin_USA_camels_14185000", "basin_USA_camels_14306500",
                  "basin_CHN_songliao_21401550", "basin_USA_camels_14400000"]
    for basin_id in basin_list:
        s3_basin_path = f's3://basins-origin/basin_shapefiles/{basin_id}.zip'
        basin_gpd = gpd.read_file(FS.open(s3_basin_path))
        basin_tz = tzfpy.get_tz(basin_gpd.geometry[0].centroid.x, basin_gpd.geometry[0].centroid.y)
        q_array = read_streamflow_from_minio(times=[['2014-12-31 17:00:00', '2019-12-31 23:00:00'],
                                          ['2020-01-01 00:00:00', '2023-12-31 23:00:00']],
                                   sta_id=basin_id.lstrip('basin_'))
        if basin_tz == 'America/Los_Angeles':
            q_array['TM'] = q_array['TM'] + np.timedelta64(7, 'h')
        elif basin_tz == 'America/Denver':
            q_array['TM'] = q_array['TM'] + np.timedelta64(6, 'h')
        elif basin_tz == 'America/Chicago':
            q_array['TM'] = q_array['TM'] + np.timedelta64(5, 'h')
        elif basin_tz == 'America/New_York':
            q_array['TM'] = q_array['TM'] + np.timedelta64(4, 'h')
        q_array_2019 = q_array[q_array['TM'] < '2020-01-01 00:00:00']
        q_array_2020 = q_array[q_array['TM'] >= '2020-01-01 00:00:00']
        # convert degree^2 to km^2
        basin_area = gpd.read_file(FS.open(s3_basin_path)).geometry[0].area * 12345.6789
        # convert ft^3 to m^3
        q_array_2020['Q'] = q_array_2020['Q'] / 35.31
        # convert m^3 to mm/h
        q_array_2020['Q'] = q_array_2020['Q'] / basin_area * 3.6
        q_array_mm_h = pd.concat([q_array_2019, q_array_2020])
        q_ds = xr.Dataset.from_dataframe(q_array_mm_h.set_index('TM'))
        q_ds.to_netcdf(basin_id+'_streamflow.nc')


def test_download_from_usgs():
    '''
    camels_hourly_csvs = '/ftproot/camels_hourly/data/usgs_streamflow_csv/'
    csv_paths = os.listdir(camels_hourly_csvs)
    minio_csv_paths = FS.glob('s3://basins-origin/basin_shapefiles/**')[1:]
    hourly_basin_ids = [path.split('-')[0] for path in csv_paths]
    minio_stcds = [csv.split('_')[-1].split('.')[0] for csv in minio_csv_paths]
    empty_stcds = [stcd for stcd in hourly_basin_ids if stcd not in minio_stcds]
    '''
    # usgs_basins = gpd.read_file("/ftproot/usgs_camels_hourly_flowdb_1373/concat_flow_camels_1373.zip")
    # first_empty_stcds = usgs_basins['basin_id'].to_numpy()
    usgs_gages = gpd.read_file("/ftproot/usgs_camels_hourly_flowdb_1373/concat_usa_usgs_all.shp")
    empty_stcds = usgs_gages['GAGE_ID'].to_numpy()
    # empty_stcds = np.union1d(first_empty_stcds, second_empty_stcds)
    for site in empty_stcds:
        site_path = f'/ftproot/usgs_camels_hourly_flowdb_1373/zq_USA_usgs_{site}.csv'
        if not os.path.exists(site_path):
            try:
                site_df = nwis.get_record(sites=site, service='iv', start='2015-01-01')
            except requests.exceptions.ConnectionError:
                continue
            except requests.exceptions.JSONDecodeError:
                continue
            site_df.to_csv(site_path)

