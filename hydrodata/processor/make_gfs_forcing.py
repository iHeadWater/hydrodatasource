import pandas as pd
import geopandas as gpd
from hydro_gistools.mean import gen_mask, mean_by_mask
import xarray as xr
import os
import numpy as np
import xarray as xr
import pandas as pd
import os
import datetime
from tqdm import tqdm
from datetime import timedelta
from hydro_opendata.reader import minio
import warnings
import scipy.interpolate

warnings.filterwarnings("ignore")
os.environ["USE_PYGEOS"] = "0"

def generate_forecast_times_updated(date_str, hour_str, num):
    # Parse the given date and hour
    date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d")
    given_hour = int(hour_str)

    # Define the forecasting hours
    forecast_hours = [0, 6, 12, 18]

    # Find the closest forecast hour before the given hour
    closest_forecast_hour = max([hour for hour in forecast_hours if hour <= given_hour])

    # Generate the forecast times
    forecast_times = []
    remaining_num = num
    while remaining_num > 0:
        time_difference = given_hour - closest_forecast_hour
        for i in range(time_difference, 6):
            if remaining_num == 0:
                break
            forecast_times.append(
                [
                    date_obj.strftime("%Y-%m-%d"),
                    str(closest_forecast_hour).zfill(2),
                    str(i).zfill(2),
                ]
            )
            remaining_num -= 1

        # Move to the next forecasting hour
        if closest_forecast_hour == 18:
            date_obj += timedelta(days=1)
            closest_forecast_hour = 0
        else:
            closest_forecast_hour += 6
        given_hour = closest_forecast_hour

    return forecast_times

def fetch_latest_data(
    date_np=np.datetime64("2022-09-01"), time_str="00", bbbox=(-125, 25, -66, 50), num=3, var_name='tp'
):
    forecast_times = generate_forecast_times_updated(date_np, time_str, num)

    gfs_reader = minio.GFSReader()
    gfs_reader.set_default_variable(var_name)
    time = forecast_times[0]
    data = gfs_reader.open_dataset(
        # data_variable="tp",
        creation_date=np.datetime64(time[0]),
        creation_time=time[1],
        bbox=bbbox,
        dataset="wis",
        time_chunks=24,
    )
    
    if isinstance(data, xr.DataArray):
        data = data.to_dataset()
    
    if "downward_shortwave_radiation_flux" in data.data_vars:
        data  = data.rename({'downward_shortwave_radiation_flux': 'dswrf'})
    if 'precipitable_water_entire_atmosphere' in data.data_vars:
        data  = data.rename({'precipitable_water_entire_atmosphere': 'pwat'})
    if "relative_humidity_2m_above_ground" in data.data_vars:
        data  = data.rename({'relative_humidity_2m_above_ground': '2r'})
    if 'specific_humidity_2m_above_ground' in data.data_vars:
        data  = data.rename({'specific_humidity_2m_above_ground': '2sh'})
    if 'temperature_2m_above_ground' in data.data_vars:
        data  = data.rename({'temperature_2m_above_ground': '2t'})
    if "total_cloud_cover_entire_atmosphere" in data.data_vars:
        data  = data.rename({'total_cloud_cover_entire_atmosphere': 'tcc'})
    if "total_precipitation_surface" in data.data_vars:
        data  = data.rename({'total_precipitation_surface': 'tp'})
    if 'u_component_of_wind_10m_above_ground' in data.data_vars:
        data  = data.rename({'u_component_of_wind_10m_above_ground': '10u'})
    if 'v_component_of_wind_10m_above_ground' in data.data_vars:
        data  = data.rename({'v_component_of_wind_10m_above_ground': '10v'})
    
    data = data[var_name].isel(valid_time=int(time[2]))
    # print(data.dims)
    if "step" in data.dims:
        data = data.max(dim="step")  
    if "time" in data.dims:
        data = data.max(dim="time")  
    
    data = data.rename({"valid_time": "time"})
    
    # print(data)
    latest_data = data
    # print(latest_data)

    for time in forecast_times[1:]:
        data = gfs_reader.open_dataset(
            # data_variable="tp",
            creation_date=np.datetime64(time[0]),
            creation_time=time[1],
            bbox=bbbox,
            dataset="wis",
            time_chunks=24,
        )
        
        if isinstance(data, xr.DataArray):
            data = data.to_dataset()
        
        if "downward_shortwave_radiation_flux" in data.data_vars:
            data  = data.rename({'downward_shortwave_radiation_flux': 'dswrf'})
        if 'precipitable_water_entire_atmosphere' in data.data_vars:
            data  = data.rename({'precipitable_water_entire_atmosphere': 'pwat'})
        if "relative_humidity_2m_above_ground" in data.data_vars:
            data  = data.rename({'relative_humidity_2m_above_ground': '2r'})
        if 'specific_humidity_2m_above_ground' in data.data_vars:
            data  = data.rename({'specific_humidity_2m_above_ground': '2sh'})
        if 'temperature_2m_above_ground' in data.data_vars:
            data  = data.rename({'temperature_2m_above_ground': '2t'})
        if "total_cloud_cover_entire_atmosphere" in data.data_vars:
            data  = data.rename({'total_cloud_cover_entire_atmosphere': 'tcc'})
        if "total_precipitation_surface" in data.data_vars:
            data  = data.rename({'total_precipitation_surface': 'tp'})
        if 'u_component_of_wind_10m_above_ground' in data.data_vars:
            data  = data.rename({'u_component_of_wind_10m_above_ground': '10u'})
        if 'v_component_of_wind_10m_above_ground' in data.data_vars:
            data  = data.rename({'v_component_of_wind_10m_above_ground': '10v'})
        
        data = data[var_name].isel(valid_time=int(time[2]))
        # data = data.squeeze(dim='step', drop=True)
        if "step" in data.dims:
            data = data.max(dim="step")   
        if "time" in data.dims:
            data = data.max(dim="time") 
        data = data.rename({"valid_time": "time"})
        latest_data = xr.concat([latest_data, data], dim="time")
        # print(latest_data)

    latest_data = latest_data.expand_dims({"time": 1}, axis=0)
    latest_data = latest_data.to_dataset()
    latest_data = latest_data.transpose("time", "lon", "lat")
    # print(latest_data)
    return latest_data

def make_gfs_dataset(
    start_time,
    end_time,
    # shp_path,
    gfs_mask_path,
    gfs_final_path,
    var_name):

    # 获取.nc(mask)文件列表
    nc_files = sorted(
        [
            os.path.join(gfs_mask_path, f)
            for f in os.listdir(gfs_mask_path)
            if f.endswith(".nc")
        ]
    )
    for nc_file in nc_files:
        nc_file_name = os.path.splitext(os.path.basename(nc_file))[0] + ".nc"
        gfs_mask_path = os.path.join(gfs_mask_path, nc_file_name)
        mask = xr.open_dataset(gfs_mask_path)
        box = (
            mask.coords["lon"][0] - 0.2,
            mask.coords["lat"][0] - 0.2,
            mask.coords["lon"][-1] + 0.2,
            mask.coords["lat"][-1] + 0.2,
        )
        w_data = mask["w"]
        # 初始日期和时间
        # start_date = datetime.datetime(2020, 7, 1)
        start_date = start_time
        # end_date = datetime.datetime(2020, 9, 30)
        end_date = end_time
        current_date_time = start_date

        # days_to_loop = 1
        # end_date = start_date + timedelta(days=days_to_loop)
        # current_date_time = start_date

        while current_date_time < end_date:
            # print(current_date_time[0])
            date_str = current_date_time.strftime("%Y-%m-%d")
            time_str = current_date_time.strftime("%H")

            print(f"Date: {date_str}, Time: {time_str}")
            # num=25意味着生成25小时
            merge_gfs_data = fetch_latest_data(
                date_np=date_str, time_str=time_str, bbbox=box, num=1, var_name=var_name
            )
            # print(merge_gfs_data)
            w_data_interpolated = w_data.interp(
                lat=merge_gfs_data.lat, lon=merge_gfs_data.lon, method="nearest"
            ).fillna(0)

            # 将 w 数据广播到与当前数据集相同的时间维度上
            w_data_broadcasted = w_data_interpolated.broadcast_like(
                merge_gfs_data[var_name]
            )
            merge_gfs_data[var_name] = merge_gfs_data[var_name] * w_data_broadcasted
            # merge_gfs_data = merge_gfs_data.rename({'tp': '2t'})
            merge_gfs_data.to_netcdf(
                gfs_final_path + date_str + " " + time_str + ":00:00.nc"
            )

            # 增加一个小时
            current_date_time += timedelta(hours=1)

# Define a function to parse datetime from filename
def parse_datetime_from_filename(filename):
    date_str = os.path.basename(filename).split('.')[0]
    return datetime.datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')

def with_time_now(input_file):
    # 1. 从文件名中解析日期和时间，并计算目标时间范围
    filename_date_str = os.path.basename(input_file).split('.')[0]
    date_time_obj = datetime.datetime.strptime(filename_date_str, '%Y-%m-%d %H:%M:%S')
    
    # 确保仅处理2017-01-01 08:00:00之后的文件
    '''
    if date_time_obj < datetime.datetime(2022, 11, 8, 0):
        return None
    '''
    # print(date_time_obj)
    start_time = date_time_obj - datetime.timedelta(hours=168)
    end_time = date_time_obj - datetime.timedelta(hours=0)
    # start_time = datetime.datetime.strptime(filename_date_str, '%Y-%m-%d %H:%M:%S')
    # end_time = datetime.datetime.strptime(filename_date_str, '%Y-%m-%d %H:%M:%S')
    # print(start_time)
    # 2. 从 gpm.nc 文件中获取对应时间范围的数据
    # subset_gpm = gpm_dataset.sel(time=slice(start_time, end_time))
    
    # 3. 打开输入文件
    ds_input = xr.open_dataset(input_file)
    # print(ds_input)
    # ds_input = ds_input.sel(time = ds_input.time[1:])
    
    # 4. 将两个文件中的变量名都更改为 "waterlevel"
    # subset_gpm = subset_gpm.rename({'__xarray_dataarray_variable__': 'tp'})
    # ds_input = ds_input.rename({'__xarray_dataarray_variable__': 'tp'})
    
    # 5. 使用 xarray 进行插值，将数据集插值到共同的 lat 和 lon 网格上
    # ds_input_interp = ds_input.interp(lat=gpm_dataset.lat, lon=gpm_dataset.lon, method='linear')
    # print(ds_input_interp)
    
    # 6. 沿 time 维度拼接两个数据集
    # combined_data = xr.concat([subset_gpm, ds_input_interp], dim='time')
    combined_data = ds_input
    # 7. 添加 time_now 维度，并将文件代表的时间设置为该维度的值
    combined_data['time_now'] = date_time_obj
    combined_data = combined_data.assign_coords({"time_now": date_time_obj})
    combined_data = combined_data.expand_dims('time_now')
    
    return combined_data

def gen_gfs(gfs_path, t_path):
    # List all the .nc files in the directory
    directory_path = gfs_path
    # directory_path = '/home/xushuolong1/flood_data_preprocess/GPM_data_preprocess/test_gfs_gpm/'

    nc_files = [f for f in os.listdir(directory_path) if f.endswith('.nc') and "gpm" not in f]

    # Sort nc_files by datetime in filename
    nc_files = sorted(nc_files, key=parse_datetime_from_filename)   
    processed_datasets_time_now = []
    for file in nc_files:
        filepath = os.path.join(directory_path, file)
        # processed_data = with_time_now(filepath)
        processed_data = xr.open_dataset(filepath)
        # processed_data.to_netcdf("data/output/" + str(file))
        # processed_data.to_netcdf('for_gpt.nc')
        # print(processed_data)
        time_values = processed_data['time'].values
        unique_time_values, counts = np.unique(time_values, return_counts=True)
        duplicate_times = unique_time_values[counts > 1]
        if len(duplicate_times) != 0:
            print(duplicate_times)
        if processed_data:
            processed_datasets_time_now.append(processed_data)

    # Combine all processed datasets
    final_combined_dataset_time_now = xr.concat(processed_datasets_time_now, dim = 'time')
    # Save the final dataset to a new .nc file
    output_file_path = t_path
    final_combined_dataset_time_now.to_netcdf(output_file_path)
    return final_combined_dataset_time_now

def gfs_forcing(start_time, end_time):
    dss = []
    # watershed_files = ['data/shp/02051500.shp']
    # for i in watershed_files:
    #     watershed = gpd.read_file(i)
    #     gen_mask(watershed, "STAID", "gfs", save_dir="data/mask")
        # gen_mask(watershed, "STAID", "gpm", save_dir="data/mask")
        # gen_mask(watershed, "STAID", "era5", save_dir="data/mask")
    gfs_mask_path = "/home/jiaxuwu/fifth/data/mask/gfs_mask/"
    attr_list = ['dswrf', 'pwat', '2r', '2sh', '2t', 'tcc', 'tp', '10u', '10v']
    start_time = start_time
    # 左闭右开
    end_time = end_time
    for i in attr_list:
        gfs_final_path = '/ftproot/camels_hourly/attr_gfs/'+i+'_gfs/'
        # 检查文件夹是否存在
        if not os.path.exists(gfs_final_path):
            # 如果不存在，则创建文件夹
            os.makedirs(gfs_final_path)
            print(f"文件夹 {gfs_final_path} 已创建。")
        else:
            print(f"文件夹 {gfs_final_path} 已存在。")
        t_path = '/ftproot/camels_hourly/attr_gfs/gfs_total/'+i+'.nc'
        
        make_gfs_dataset(
            start_time = start_time,
            end_time = end_time,
            # shp_path = shp_path,
            gfs_mask_path = gfs_mask_path,
            gfs_final_path = gfs_final_path,
            var_name=i
        )

        ds = gen_gfs(
            gfs_path=gfs_final_path, 
            t_path=t_path
        )
        dss.append(ds)
    
    combined_ds = xr.merge(dss)
    combined_ds.to_netcdf('/ftproot/camels_hourly/attr_gfs/gfs_total/attr_gfs.nc')
    
def interpolate_gfs(file_path, new_file_path, var_name):
    # 读取NetCDF文件
    data = xr.open_dataset(file_path)

    # 新的纬度和经度值
    new_lat = np.array([36.65, 36.75, 36.85, 36.95, 37.05])
    new_lon = np.array([-78.15, -78.05, -77.95, -77.85, -77.75])

    # 提取原始纬度、经度和变量 '2r'
    original_lat = data['lat'].values
    original_lon = data['lon'].values
    variable_2r = data[var_name].values

    # 调整 '2r' 变量的维度顺序
    variable_2r_transposed = np.transpose(variable_2r, (0, 2, 1))  # 转置为 (time, lat, lon)

    # 为每个时间点执行双线性插值
    interpolated_2r = np.empty((len(data['time']), len(new_lon), len(new_lat)))
    for t in range(len(data['time'])):
        interp_func = scipy.interpolate.interp2d(original_lon, original_lat, variable_2r_transposed[t, :, :], kind='linear')
        interpolated_2r[t, :, :] = interp_func(new_lon, new_lat)

    # 创建一个新的xarray Dataset来保存插值后的数据
    interpolated_ds = xr.Dataset(
        {
            var_name: (("time", "lon", "lat"), interpolated_2r)
        },
        coords={
            "time": data["time"],
            "lat": new_lat,
            "lon": new_lon
        }
    )
    # 保存新的数据集为NetCDF文件
    interpolated_ds.to_netcdf(new_file_path)
    # 关闭原始数据文件
    data.close()

# gfs_forcing(start_time = datetime.datetime(2017, 1, 1), end_time = datetime.datetime(2022, 12, 31))
