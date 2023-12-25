# pytest model_stream.py::test_auto_stream
import os.path

import numpy as np
import pandas as pd
import pytest
import torch
import urllib3 as ur
from yaml import load, Loader

import minio_api as ma
from hydroprivatedata import config
import pathlib as pl
import smtplib
import email

work_dir = pl.Path(os.path.abspath(os.curdir)).parent


@pytest.mark.asyncio
async def test_auto_stream():
    model_name = 'dt_reg_test'
    client = config.mc
    file_names = test_read_history('aiff', '001')
    model_info = file_names[model_name]
    await ma.minio_download_csv(client=client, bucket_name='models', object_name=model_info, file_path=model_info)
    # 下面的model_info是文件路径
    model = torch.load(model_info)
    result = model.predict(test_read_valid_data())
    # rmse、r2、kge等等
    # https://zhuanlan.zhihu.com/p/631317974


def test_read_history(user_model_type, version):
    history_dict_path = os.path.relpath('test_data/history_dict.pkl')
    # 姑且假设所有模型都被放在test_data/models文件夹下
    if not os.path.exists(history_dict_path):
        history_dict = {}
        models = os.listdir(os.path.relpath('test_data/models'))
        # 姑且假设model的名字为floodforecast_v1.pth，即用途_版本.pth
        model_vers = [int(model.split('.')[0].split('v')[1]) for model in models]
        max_version = np.max(model_vers)
        model_file_name = user_model_type+'_v'+str(version)+'.pth'
        if model_file_name in models:
            history_dict[user_model_type] = max_version
            np.save(history_dict_path, history_dict, allow_pickle=True)
        return history_dict
    else:
        history_dict = np.load(history_dict_path, allow_pickle=True)
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
    # start_time = datetime.datetime.strptime(test_period[0]+' 00:00:00', '%Y-%m-%d %H:%M:%S')
    start_time = pd.to_datetime(test_period[0]+' 00:00:00', format='%Y-%m-%d %H:%M:%S').tz_localize(tz='UTC')
    # end_time = datetime.datetime.strptime(test_period[1]+' 00:00:00', '%Y-%m-%d %H:%M:%S')
    end_time = pd.to_datetime(test_period[1]+' 00:00:00', format='%Y-%m-%d %H:%M:%S').tz_localize(tz='UTC')
    obj_list = np.array([obj.object_name for obj in client_mc.list_objects(bucket_name='forestbat-private', recursive='True')])
    # https://stackoverflow.com/questions/71050211/typeerror-invalid-comparison-between-dtype-datetime64ns-utc-and-datetime64
    obj_time_list = pd.to_datetime([obj.last_modified for obj in client_mc.list_objects(bucket_name='forestbat-private', recursive='True')])
    time_indexes = obj_time_list[(obj_time_list > start_time) & (obj_time_list < end_time)]
    obj_down_array = obj_list[time_indexes]
    await ma.minio_batch_download(obj_down_array, client_mc, bucket_name='forestbat-private', local_path='test_data')
    return obj_down_array
