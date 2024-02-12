#!/home/zhujianfeng/.conda/envs/stac-test/bin/python

import datetime
import json
import os
import subprocess
from pathlib import Path

import kerchunk.grib2
import numpy as np
import s3fs
import ujson
from apscheduler.schedulers.blocking import BlockingScheduler  # 后台运行
from kerchunk.combine import MultiZarrToZarr, drop
from minio import Minio
import yaml

work_dir = Path(__file__).resolve().parent
with open(work_dir / 'privacy_config.yaml', 'r') as f:
    privacy_config = yaml.safe_load(f)
storage_options = {
    'client_kwargs': {'endpoint_url': privacy_config['minio']['client_endpoint']},
    'key': privacy_config['minio']['access_key'],
    'secret': privacy_config['minio']['secret']
}
minioClient = Minio(storage_options['client_kwargs'],
                    storage_options['key'],
                    secret_key=privacy_config['minio']['secret'],
                    secure=False)
fs = s3fs.S3FileSystem(
    client_kwargs={"endpoint_url": privacy_config['minio']['client_endpoint']},
    key=privacy_config['minio']['access_key'],
    secret=privacy_config['minio']['secret']
)

sc = BlockingScheduler(timezone="Asia/Shanghai")

bbox = [73, 3, 136, 54]


@sc.scheduled_job('cron', day_of_week='*', hour='12', minute='10', second='00')
def gfs_downloader_00():
    f = open('log.txt', 'a', encoding='utf8')
    date = datetime.datetime.now()
    # date=date + datetime.timedelta(days = -1)
    date = date.strftime('%Y%m%d')
    creation_time = '00'
    urls = []
    # https://nomads.ncep.noaa.gov/gribfilter.php?ds=gfs_0p25_1hr
    # https://github.com/albertotb/get-gfs
    # https://dtcenter.org/nwp-containers-online-tutorial/publicly-available-data-sets
    # https://www.nco.ncep.noaa.gov/pmb/docs/grib2/grib2_doc/, PWAT指降雨
    url_ = ('https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?file=gfs.tCCz.pgrb2.0p25.fFFF'
            '&lev_10_m_above_ground=on&lev_2_m_above_ground=on&lev_entire_atmosphere=on&lev_entire_atmosphere_%5C'
            '%28considered_as_a_single_layer%5C%29=on&lev_surface=on&var_APCP=on&var_DSWRF=on&var_PWAT=on&var_RH=on'
            '&var_SPFH=on&var_TCDC=on&var_TMP=on&var_UGRD=on&var_VGRD=on&subregion=&leftlon=') + str(
        bbox[0]) + '&rightlon=' + str(bbox[2]) + '&toplat=' + str(bbox[3]) + '&bottomlat=' + str(
        bbox[1]) + '&dir=%2Fgfs.YYYYMMDD%2FCC%2Fatmos'
    # save_dir='gfs/'
    for forecast_time in range(1, 121):
        # ncep_gfs.get_gfs_from_ncep(date,creation_time,forecast_time,bbox=[73,3,136,54],save_dir=save_dir)
        url = url_.replace('YYYYMMDD', date).replace('CC', creation_time.zfill(2)).replace('FFF',
                                                                                           str(forecast_time).zfill(3))
        urls.append(url)

    with open('./urls.txt', mode='w') as gpm:
        [gpm.write(str(url) + '\n') for url in urls]

    subprocess.run("cat urls.txt | tr -d '\r' | xargs -n 1 curl -LJO -n -c ~/.urs_cookies -b ~/.urs_cookies",
                   shell=True)

    check_file()
    upload2server(date, creation_time)
    single_json(date, creation_time)
    multi_json(date, creation_time)
    print(datetime.datetime.now().strftime('%Y%m%d-%H:%M:%S'), '定时任务成功执行')
    print(datetime.datetime.now().strftime('%Y%m%d-%H:%M:%S'), '定时任务成功执行', file=f)


@sc.scheduled_job('cron', day_of_week='*', hour='18', minute='10', second='00')
def gfs_downloader_06():
    f = open('log.txt', 'a', encoding='utf8')
    date = datetime.datetime.now()
    date = date.strftime('%Y%m%d')
    creation_time = '06'
    urls = []
    url_ = (
               'https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?file=gfs.tCCz.pgrb2.0p25.fFFF&lev_10_m_above_ground'
               '=on&lev_2_m_above_ground=on&lev_entire_atmosphere=on'
               '&lev_entire_atmosphere_%5C%28considered_as_a_single_layer%5C%29=on&lev_surface=on&var_APCP=on&'
               'var_DSWRF=on&var_PWAT=on&var_RH=on&var_SPFH=on&var_TCDC=on&var_TMP=on&var_UGRD=on&var_VGRD=on&subregion'
               '=&leftlon=') + str(
        bbox[0]) + '&rightlon=' + str(bbox[2]) + '&toplat=' + str(bbox[3]) + '&bottomlat=' + str(
        bbox[1]) + '&dir=%2Fgfs.YYYYMMDD%2FCC%2Fatmos'

    # save_dir='gfs/'
    for forecast_time in range(1, 121):
        # ncep_gfs.get_gfs_from_ncep(date,creation_time,forecast_time,bbox=[73,3,136,54],save_dir=save_dir)
        url = url_.replace('YYYYMMDD', date).replace('CC', creation_time.zfill(2)).replace('FFF',
                                                                                           str(forecast_time).zfill(3))
        urls.append(url)
    with open('./urls.txt', mode='w') as gpm:
        [gpm.write(str(url) + '\n') for url in urls]
    subprocess.run("cat urls.txt | tr -d '\r' | xargs -n 1 curl -LJO -n -c ~/.urs_cookies -b ~/.urs_cookies",
                   shell=True)
    check_file()
    upload2server(date, creation_time)
    single_json(date, creation_time)
    multi_json(date, creation_time)
    print(datetime.datetime.now().strftime('%Y%m%d-%H:%M:%S'), '定时任务成功执行')
    print(datetime.datetime.now().strftime('%Y%m%d-%H:%M:%S'), '定时任务成功执行', file=f)
    f.close()


@sc.scheduled_job('cron', day_of_week='*', hour='00', minute='10', second='00')
def gfs_downloader_12():
    date = datetime.datetime.now()
    date = date + datetime.timedelta(days=-1)
    date = date.strftime('%Y%m%d')
    creation_time = '12'
    urls = []
    url_ = ('https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?file=gfs.tCCz.pgrb2.0p25.fFFF'
            '&lev_10_m_above_ground=on&lev_2_m_above_ground=on&lev_entire_atmosphere=on&lev_entire_atmosphere_%5C'
            '%28considered_as_a_single_layer%5C%29=on&lev_surface=on&var_APCP=on&var_DSWRF=on&var_PWAT=on&var_RH=on'
            '&var_SPFH=on&var_TCDC=on&var_TMP=on&var_UGRD=on&var_VGRD=on&subregion=&leftlon=') + str(
        bbox[0]) + '&rightlon=' + str(bbox[2]) + '&toplat=' + str(bbox[3]) + '&bottomlat=' + str(
        bbox[1]) + '&dir=%2Fgfs.YYYYMMDD%2FCC%2Fatmos'
    # save_dir='gfs/'
    for forecast_time in range(1, 121):
        # ncep_gfs.get_gfs_from_ncep(date,creation_time,forecast_time,bbox=[73,3,136,54],save_dir=save_dir)
        url = url_.replace('YYYYMMDD', date).replace('CC', creation_time.zfill(2)).replace('FFF',
                                                                                           str(forecast_time).zfill(3))
        urls.append(url)

    with open('./urls.txt', mode='w') as gpm:
        [gpm.write(str(url) + '\n') for url in urls]
    subprocess.run("cat urls.txt | tr -d '\r' | xargs -n 1 curl -LJO -n -c ~/.urs_cookies -b ~/.urs_cookies",
                   shell=True)
    check_file()
    upload2server(date, creation_time)
    single_json(date, creation_time)
    multi_json(date, creation_time)
    print(datetime.datetime.now().strftime('%Y%m%d-%H:%M:%S'), '定时任务成功执行')
    print(datetime.datetime.now().strftime('%Y%m%d-%H:%M:%S'), '定时任务成功执行', file=f)


@sc.scheduled_job('cron', day_of_week='*', hour='06', minute='10', second='00')
def gfs_downloader_18():
    date = datetime.datetime.now()
    date = date + datetime.timedelta(days=-1)
    date = date.strftime('%Y%m%d')
    creation_time = '18'
    urls = []
    url_ = ('https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?file=gfs.tCCz.pgrb2.0p25.fFFF'
            '&lev_10_m_above_ground=on&lev_2_m_above_ground=on&lev_entire_atmosphere=on&lev_entire_atmosphere_%5C'
            '%28considered_as_a_single_layer%5C%29=on&lev_surface=on&var_APCP=on&var_DSWRF=on&var_PWAT=on&var_RH=on'
            '&var_SPFH=on&var_TCDC=on&var_TMP=on&var_UGRD=on&var_VGRD=on&subregion=&leftlon=') + str(
        bbox[0]) + '&rightlon=' + str(bbox[2]) + '&toplat=' + str(bbox[3]) + '&bottomlat=' + str(
        bbox[1]) + '&dir=%2Fgfs.YYYYMMDD%2FCC%2Fatmos'
    # save_dir='gfs/'
    for forecast_time in range(1, 121):
        # ncep_gfs.get_gfs_from_ncep(date,creation_time,forecast_time,bbox=[73,3,136,54],save_dir=save_dir)
        url = url_.replace('YYYYMMDD', date).replace('CC', creation_time.zfill(2)).replace('FFF',
                                                                                           str(forecast_time).zfill(3))
        urls.append(url)
    with open('./urls.txt', mode='w') as gpm:
        [gpm.write(str(url) + '\n') for url in urls]
    subprocess.run("cat urls.txt | tr -d '\r' | xargs -n 1 curl -LJO -n -c ~/.urs_cookies -b ~/.urs_cookies",
                   shell=True)
    check_file()
    upload2server(date, creation_time)
    single_json(date, creation_time)
    multi_json(date, creation_time)
    print(datetime.datetime.now().strftime('%Y%m%d-%H:%M:%S'), '定时任务成功执行')
    print(datetime.datetime.now().strftime('%Y%m%d-%H:%M:%S'), '定时任务成功执行', file=f)


def check_file():
    while True:
        count = 0
        for root, dirs, filenames in os.walk('.'):
            for filename in filenames:
                if 'pgrb2' in filename:
                    filepath = os.path.join(root, filename)
                    size = os.path.getsize(filepath)
                    count = count + 1
                    if size < 600000:
                        os.remove(filepath)
                        count = count - 1
        if count >= 120:
            break

        else:
            subprocess.run("cat urls.txt | tr -d '\r' | xargs -n 1 curl -LJO -n -c ~/.urs_cookies -b ~/.urs_cookies",
                           shell=True)
    print("检查下载完成！")


def str_insert(str_origin, pos, str_add):
    str_list = list(str_origin)
    str_list.insert(pos, str_add)
    str_out = ''.join(str_list)
    return str_out


def upload2server(gfs_date, create_time):
    for root, dirs, filenames in os.walk('.'):
        for filename in filenames:
            if 'pgrb2' in filename:
                # gfs_date = filename[3:11]
                year = gfs_date[0:4]
                month = gfs_date[4:6]
                day = gfs_date[6:8]
                # create_time = filename[13:15]
                gfs_name = str_insert(filename, 3, gfs_date)
                minio_path = f'geodata/gfs/{year}/{month}/{day}/{create_time}/{gfs_name}'
                try:
                    print(minioClient.fput_object('watermodel-pub', minio_path, os.path.join(root, filename)))
                    path = Path(f"/ftproot/geodata/gfs/{year}/{month}/{day}/{create_time}")
                    if not path.is_dir():
                        path.mkdir(parents=True, exist_ok=True)
                    file_path = os.path.join(root, filename)
                    os.system(f"mv {file_path} /ftproot/geodata/gfs/{year}/{month}/{day}/{create_time}/{gfs_name}")
                except Exception as err:
                    print(err)
    print("上传完成！")


def gen_json(flist):
    for furl in flist:

        try:

            out = [kerchunk.grib2.scan_grib(u, storage_options=storage_options, inline_threshold=3000,
                                            filter={"shortName": "tp"}) for u in [furl]]
            outs = out[0][0]
            outf = furl + '.json'  # file name to save json to
            outf = outf.replace('watermodel-pub', 'test').replace('gfs/', 'gfs/tp/')

            with fs.open(outf, 'wb') as ff:
                ff.write(ujson.dumps(outs).encode())

            # print(outf)
        except:
            print(furl, "出错！")


def single_json(gfs_date, create_time):
    year = gfs_date[0:4]
    month = gfs_date[4:6]
    day = gfs_date[6:8]

    flist = fs.glob(f'watermodel-pub/geodata/gfs/{year}/{month}/{day}/{create_time}/*')
    flist = ['s3://' + str for str in flist]
    # print(flist)
    gen_json(flist)


def fn_to_time(index, fs, var, fn):
    import re
    import datetime
    import time
    today = datetime.datetime.strptime(time.strftime("%Y-%m-%d %H-%M-%S", time.gmtime(int(fs['time'][0]))),
                                       '%Y-%m-%d %H-%M-%S')
    ex = re.compile(r'.*f(\d+)')
    i = int(ex.match(fn).groups()[0])
    offset = datetime.timedelta(hours=i)
    return today + offset


def fn_to_index(index, fs, var, fn):
    import re
    ex = re.compile(r'.*f(\d+)')
    i = int(ex.match(fn).groups()[0])
    return i


def gen_mzz(year, month, day, time):
    json_list = fs.glob(
        f"test/geodata/gfs/tp/{str(year).zfill(4)}/{str(month).zfill(2)}/{str(day).zfill(2)}/{time}/*.json")
    if len(json_list) == 0:
        return
    json_list = ['s3://' + str for str in json_list]

    mzz = MultiZarrToZarr(
        json_list,
        target_options=storage_options,
        remote_protocol='s3',
        remote_options=storage_options,
        preprocess=drop(("surface", "step", "valid_time")),
        coo_map={'valid_time': fn_to_time, 'step': fn_to_index},
        coo_dtypes={'valid_time': np.dtype('M8[ns]'), 'step': np.dtype('int64')},
        concat_dims=['valid_time', 'step'],
        identical_dims=['latitude', 'longitude', 'time']
    )

    d = mzz.translate()
    with fs.open(
        f's3://test/geodata/gfs/tp/{str(year).zfill(4)}/{str(month).zfill(2)}/{str(day).zfill(2)}/gfs{str(year).zfill(4)}{str(month).zfill(2)}{str(day).zfill(2)}.t{time}z.0p25.json',
        'wb') as ff:
        ff.write(ujson.dumps(d).encode())
    print(
        f's3://test/geodata/gfs/tp/{str(year).zfill(4)}/{str(month).zfill(2)}/{str(day).zfill(2)}/gfs{str(year).zfill(4)}{str(month).zfill(2)}{str(day).zfill(2)}.t{time}z.0p25.json')

    with fs.open('test/geodata/gfs/gfs.json') as ff:
        cont = json.load(ff)
    cont['tp'][-1]['end'] = f'{year}-{month}-{day}T{time}'
    with fs.open('test/geodata/gfs/gfs.json', 'w') as ff:
        json.dump(cont, ff)
    print('更新至', f'{year}-{month}-{day}T{time}')


def multi_json(gfs_date, create_time):
    year = gfs_date[0:4]
    month = gfs_date[4:6]
    day = gfs_date[6:8]
    gen_mzz(year, month, day, create_time)


if __name__ == '__main__':
    try:
        sc.start()
    except Exception as e:
        sc.shutdown()
        print(datetime.datetime.now().strftime('%Y%m%d-%H:%M:%S'), '下载任务停止')
