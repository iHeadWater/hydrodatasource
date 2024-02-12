# pytest model_stream.py::test_auto_stream
import os.path

import numpy as np
import pandas as pd
import pytest
import torch
import urllib3 as ur
from yaml import load, Loader

import minio_api as ma
from hydrodata import config
import pathlib as pl
import smtplib
import email
import xarray as xr

work_dir = pl.Path(os.path.abspath(os.curdir)).parent


def test_nc2tensor(nc_path='test_data/gpm_gfs_2019_01_01_00_2019_01_31_23.nc'):
    nc_ds = xr.open_dataset(os.path.join(work_dir, nc_path))
    nc_df = nc_ds.to_dataframe()


@pytest.mark.asyncio
async def test_auto_stream():
    # test_data = await test_read_valid_data('001')
    test_model_name = test_read_history(user_model_type='model', version='300')
    test_model_path = os.path.join(work_dir, 'test_data/models', 'model_v'+test_model_name['model']+'.pth')
    # 下面的model_info是文件路径
    model = torch.load(test_model_path, map_location=torch.device('cpu'))
    model
    # result = model.predict(test_data)
    # rmse、r2、kge等等
    # https://zhuanlan.zhihu.com/p/631317974


def test_read_history(user_model_type='wasted', version='1'):
    history_dict_path = os.path.join(work_dir, 'test_data/history_dict.npy')
    # 姑且假设所有模型都被放在test_data/models文件夹下
    if not os.path.exists(history_dict_path):
        history_dict = {}
        models = os.listdir(os.path.join(work_dir, 'test_data/models'))
        # 姑且假设model的名字为wasted_v1.pth，即用途_版本.pth
        current_max = 0
        for model_name in models:
            if user_model_type in model_name:
                model_ver = int(model_name.split('.')[0].split('v')[1])
                if model_ver > current_max:
                    current_max = model_ver
        model_file_name = user_model_type+'_v'+str(version)+'.pth'
        if model_file_name in models:
            history_dict[user_model_type] = current_max
            np.save(history_dict_path, history_dict, allow_pickle=True)
        return history_dict
    else:
        history_dict = np.load(history_dict_path, allow_pickle=True).flatten()[0]
        model_file_name = user_model_type + '_v' + str(version) + '.pth'
        if model_file_name not in history_dict.keys():
            history_dict[user_model_type] = version
        return history_dict


@pytest.mark.asyncio
async def test_read_valid_data(version='001'):
    client_mc = config.mc
    config_path = os.path.join(work_dir, 'test_data/aiff_config/aiff_v'+str(version)+'.yml')
    if not os.path.exists(config_path):
        version_url = 'https://raw.githubusercontent.com/iHeadWater/AIFloodForecast/main/scripts/conf/v'+str(version)+'.yml'
        yml_str = ur.request('GET', version_url).data.decode('utf8')
    else:
        with open(config_path, 'r') as fp:
            yml_str = fp.read()
    conf_yaml = load(yml_str, Loader=Loader)
    test_period = conf_yaml['test_period']
    # test_period = ["2023-11-01 16:38:00", "2023-11-01 16:40:00"]
    start_time = pd.to_datetime(test_period[0], format='%Y-%m-%d %H:%M:%S').tz_localize(tz='Asia/Shanghai')
    end_time = pd.to_datetime(test_period[1], format='%Y-%m-%d %H:%M:%S').tz_localize(tz='Asia/Shanghai')
    obj_time_list = pd.to_datetime([obj.last_modified for obj in client_mc.list_objects(bucket_name='forestbat-private', recursive='True')]).tz_convert(tz='Asia/Shanghai')
    time_indexes = obj_time_list[(obj_time_list > start_time) & (obj_time_list < end_time)]
    obj_down_array = [obj.object_name for obj in client_mc.list_objects(bucket_name='forestbat-private', recursive='True') if obj.last_modified in time_indexes]
    await ma.minio_batch_download(obj_down_array, client_mc, bucket_name='forestbat-private', local_path=os.path.join(work_dir, 'test_data'))
    return obj_down_array
