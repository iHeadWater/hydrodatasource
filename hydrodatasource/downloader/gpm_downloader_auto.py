#!/home/zhujianfeng/.conda/envs/stac-test/bin/python

import calendar
import datetime
import subprocess
from pathlib import Path

import s3fs
import yaml

from apscheduler.schedulers.blocking import BlockingScheduler  # 后台运行
from minio import Minio


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

flist = []

sc = BlockingScheduler(timezone="Asia/Shanghai")


@sc.scheduled_job('cron', day_of_week='*', hour='*', minute='55', second='00')
def gpm_downloader():
    # f = open('log.txt', 'a', encoding='utf8')

    date = datetime.datetime.now()
    date = date + datetime.timedelta(hours=-14)
    # date=date.strftime('%Y%m%d')

    year = int(date.strftime('%Y'))
    month = int(date.strftime('%m'))
    day = int(date.strftime('%d'))
    hour = int(date.strftime('%H'))

    new = []
    date = str(year) + str(month).zfill(2) + str(day).zfill(2)

    for time in range(0, 2):
        # time=time*30
        # should.append(date+str(time).zfill(4))

        # url='https://gpm1.gesdisc.eosdis.nasa.gov/daac-bin/OTF/HTTP_services.cgi?FILENAME=%2Fdata%2FGPM_L3%2FGPM_3IMERGHHE.06%2FYYYY%2FAAA%2F3B-HHR-E.MS.MRG.3IMERG.YYYYMMDD-SBBBBBB-ECCCCCC.EEEE.V06C.HDF5&FORMAT=bmM0Lw&BBOX=3%2C73%2C54%2C136&LABEL=3B-HHR-E.MS.MRG.3IMERG.YYYYMMDD-SBBBBBB-ECCCCCC.EEEE.V06C.HDF5.SUB.nc4&SHORTNAME=GPM_3IMERGHHE&SERVICE=L34RS_GPM&VERSION=1.02&DATASET_VERSION=06&VARIABLES=precipitationCal%2CrandomError%2CprecipitationUncal%2CIRkalmanFilterWeight%2CHQprecipSource%2CHQprecipitation%2CprobabilityLiquidPrecipitation%2CHQobservationTime%2CIRprecipitation'
        # url='https://gpm1.gesdisc.eosdis.nasa.gov/opendap/GPM_L3/GPM_3IMERGHHE.06/YYYY/AAA/3B-HHR-E.MS.MRG.3IMERG.YYYYMMDD-SBBBBBB-ECCCCCC.EEEE.V06D.HDF5.nc4?IRkalmanFilterWeight%5B0:0%5D%5B2530:3159%5D%5B930:1439%5D,HQprecipSource%5B0:0%5D%5B2530:3159%5D%5B930:1439%5D,precipitationCal%5B0:0%5D%5B2530:3159%5D%5B930:1439%5D,precipitationUncal%5B0:0%5D%5B2530:3159%5D%5B930:1439%5D,HQprecipitation%5B0:0%5D%5B2530:3159%5D%5B930:1439%5D,probabilityLiquidPrecipitation%5B0:0%5D%5B2530:3159%5D%5B930:1439%5D,HQobservationTime%5B0:0%5D%5B2530:3159%5D%5B930:1439%5D,randomError%5B0:0%5D%5B2530:3159%5D%5B930:1439%5D,IRprecipitation%5B0:0%5D%5B2530:3159%5D%5B930:1439%5D,time,lon%5B2530:3159%5D,lat%5B930:1439%5D'
        # url='https://gpm1.gesdisc.eosdis.nasa.gov/daac-bin/OTF/HTTP_services.cgi?FILENAME=%2Fdata%2FGPM_L3%2FGPM_3IMERGHHE.06%2F2023%2FAAA%2F3B-HHR-E.MS.MRG.3IMERG.YYYYMMDD-SBBBBBB-ECCCCCC.EEEE.V06D.HDF5&DATASET_VERSION=06&SERVICE=L34RS_GPM&FORMAT=bmM0Lw&SHORTNAME=GPM_3IMERGHHE&VERSION=1.02&VARIABLES=precipitationCal%2CrandomError%2CprecipitationUncal%2CIRkalmanFilterWeight%2CHQprecipSource%2CHQprecipitation%2CprobabilityLiquidPrecipitation%2CHQobservationTime%2CIRprecipitation&LABEL=3B-HHR-E.MS.MRG.3IMERG.YYYYMMDD-SBBBBBB-ECCCCCC.EEEE.V06D.HDF5.SUB.nc4&BBOX=3%2C73%2C54%2C136'
        url = ('https://gpm1.gesdisc.eosdis.nasa.gov/daac-bin/OTF/HTTP_services.cgi?'
               'FILENAME=%2Fdata%2FGPM_L3%2FGPM_3IMERGHHE.06%2FYYYY%2FAAA%2F3B-HHR-E.MS.MRG.3IMERG.YYYYMMDD-SBBBBBB'
               '-ECCCCCC.EEEE.V06E.HDF5&VARIABLES=precipitationCal%2CrandomError%2CprecipitationUncal'
               '%2CIRkalmanFilterWeight%2CHQprecipSource%2CHQprecipitation%2CprobabilityLiquidPrecipitation'
               '%2CHQobservationTime%2CIRprecipitation&DATASET_VERSION=06&BBOX=3%2C73%2C54%2C136&SERVICE=L34RS_GPM'
               '&VERSION=1.02&SHORTNAME=GPM_3IMERGHHE&FORMAT=bmM0Lw&LABEL=3B-HHR-E.MS.MRG.3IMERG.YYYYMMDD-SBBBBBB'
               '-ECCCCCC.EEEE.V06E.HDF5.SUB.nc4')

        YYYY = str(year)
        MM = str(month).zfill(2)
        DD = str(day).zfill(2)

        AAA = str(daynum(year, month, day)).zfill(3)

        EEEE = str(hour * 60 + time * 30).zfill(4)

        start = int(hour * 10000 + time * 30 * 100)
        BBBBBB = str(start).zfill(6)

        end = int(start + 2959)
        CCCCCC = str(end).zfill(6)

        url = url.replace('YYYY', YYYY).replace('MM', MM).replace('DD', DD).replace('AAA', AAA).replace('EEEE',
                                                                                                        EEEE).replace(
            'BBBBBB', BBBBBB).replace('CCCCCC', CCCCCC)
        # print(url)
        # break
        new.append(url)

    with open('./gpm.txt', mode='w') as gpm:
        [gpm.write(str(url) + '\n') for url in new]

    subprocess.run("cat gpm.txt | tr -d '\r' | xargs -n 1 curl -LJO -n -c ~/.urs_cookies -b ~/.urs_cookies", shell=True)

    flist = []
    upload()

    update()

    print(datetime.datetime.now().strftime('%Y%m%d-%H:%M:%S'), '定时任务成功执行')
    # print(datetime.datetime.now().strftime('%Y%m%d-%H:%M:%S'),'定时任务成功执行',file=f)

    # f.close()


def daynum(year, month, day):
    totalday = 0
    if calendar.isleap(year):
        days = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    else:
        days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    for i in range(1, 13):
        if i == month:
            for j in range(1, i):
                totalday = totalday + days[j - 1]
    totalday = totalday + day
    return totalday


def upload():
    for root, dirs, filenames in os.walk('.'):
        for filename in filenames:

            if ".nc4" in filename and "2024" in filename:

                gpm_date = filename[23:31]
                year = gpm_date[0:4]
                month = gpm_date[4:6]
                day = gpm_date[6:8]

                minio_path = f'geodata/gpm/{year}/{month}/{day}/{filename}'

                # print(minio_path)

                # # minio_path = minio_path.replace('_land','')

                try:
                    print(minioClient.fput_object('watermodel-pub', minio_path, os.path.join(root, filename)))
                    flist.append(minio_path)
                    if not os.path.exists(f"/ftproot/geodata/gpm_30m/{year}"):
                        os.mkdir(f"/ftproot/geodata/gpm_30m/{year}")
                    os.system(f"mv {filename} /ftproot/geodata/gpm_30m/{year}")
                except Exception as err:
                    print(err)


import kerchunk.hdf

import os
import ujson
import json

so = dict(mode='rb', storage_options=storage_options, default_fill_cache=False,
          default_cache_type='first')  # args to fs.open()


# default_fill_cache=False avoids caching data in between file chunks to lowers memory usage.

def gen_json(file_url):
    with fs.open(file_url, **so) as infile:
        # print(file_url)
        try:
            h5chunks = kerchunk.hdf.SingleHdf5ToZarr(infile, file_url)
        except:
            f = open('zarrmissed.txt', mode='a')
            print(file_url, file=f)
            f.close()
            return

        outf = file_url[:-4] + '.json'  # file name to save json to
        outf = outf.replace('watermodel-pub', 'test')
        with fs.open(outf, 'wb') as f:
            f.write(ujson.dumps(h5chunks.translate()).encode());


def single_json(nc_list):
    for file in nc_list:
        gen_json(file)


from kerchunk.combine import MultiZarrToZarr


def multi_json(flist):
    # print('multi_json')

    flist.sort()
    lasted = flist[-1]

    filename = lasted.split('/')[-1]

    gpm_date = filename[23:39]
    yyyy = gpm_date[0:4]
    mm = gpm_date[4:6]
    dd = gpm_date[6:8]
    hh = gpm_date[10:12]
    mi = gpm_date[12:14]
    ss = gpm_date[14:16]

    json_list = fs.glob(f"test/geodata/gpm/{yyyy}/{mm}/*/*.json")
    json_list = ['s3://' + str for str in json_list]

    mzz = MultiZarrToZarr(
        json_list,
        target_options=storage_options,
        remote_protocol='s3',
        remote_options=storage_options,
        concat_dims=['time'],
        identical_dims=['lat', 'lon']
    )

    d = mzz.translate()

    # output = f's3://test/geodata/gpm/{yyyy}/{mm}/gpm{yyyy}{mm}_{dd}.json'
    output = f's3://test/geodata/gpm/{yyyy}/{mm}/gpm{yyyy}{mm}_inc.json'
    # days = calendar.monthrange(int(yyyy),int(mm))[1]
    # if int(dd)==days and hh=='23' and mi=='30':
    #     output = f's3://test/geodata/gpm/{yyyy}/{mm}/gpm{yyyy}{mm}_inc.json'
    with fs.open(output, 'wb') as f:
        f.write(ujson.dumps(d).encode())
    print(output, "写入成功！")

    with fs.open('test/geodata/gpm/gpm.json') as f:
        cont = json.load(f)
    cont['end'] = f'{yyyy}-{mm}-{dd}T{hh}:{mi}:00.000000000'
    with fs.open('test/geodata/gpm/gpm.json', 'w') as f:
        json.dump(cont, f)
    print('更新至', filename)


def update():
    nc_list = ['watermodel-pub/' + str for str in flist]
    single_json(nc_list)

    multi_json(flist)


if __name__ == '__main__':

    try:
        sc.start()

    except Exception as e:
        sc.shutdown()
        print(datetime.datetime.now().strftime('%Y%m%d-%H:%M:%S'), '下载任务停止')
        # print(datetime.datetime.now().strftime('%Y%m%d-%H:%M:%S'),'下载任务停止',file=f)
