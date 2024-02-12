import pandas as pd
import xarray as xr
import numpy as np

def concat_csv():
    file_paths = ['/batch01.csv', '/batch02.csv', '/batch03.csv', '/batch04.csv', '/batch05.csv', '/batch06.csv',
                '/batch07.csv', '/batch08.csv']
    paths = ['21401550', '02051500']

    for i in paths:
        dfs = []
        for j in file_paths:
            # 指定要合并的文件的路径
            file_path = 'data/'+i+j
            # 读取文件
            df = pd.read_csv(file_path)
            dfs.append(df)
        
        # 合并 DataFrame
        combined_df = pd.concat(dfs, ignore_index=True)

        # 保存合并后的 DataFrame 到新的 CSV 文件
        output_file_path = 'data/'+i+'/'+i+'.csv'
        combined_df.to_csv(output_file_path, index=False)

        # 输出文件路径以供下载
        print(output_file_path)

def rename_df():
    df = pd.read_csv("data/02051500/02051500.csv")
    df.rename(columns={'FID': 'STAID', '.geo':'geo'}, inplace=True)
    df['STAID']=['02051500']*len(df)
    df.to_csv('data/02051500/02051500.csv', index=False)

def csv_nc():
    # 读取 csv 文件
    file_path = 'data/02051500/02051500.csv'  # 替换为您的文件路径
    df = pd.read_csv(file_path)
    df.rename(columns={'system:index':'TM'}, inplace=True)
    df['TM'] = pd.to_datetime(df['TM'], format='%Y%m%dT%H_00000000000000000000')
    basin_value = '02051500'
    # 创建一个新的 xarray 数据集
    ds = xr.Dataset(
        {
            "dewpoint_temperature_2m": (("time", "basin"), df[['dewpoint_temperature_2m']].values),
            "potential_evaporation": (("time", "basin"), df[['potential_evaporation']].values),
            "snow_depth_water_equivalent": (("time", "basin"), df[['snow_depth_water_equivalent']].values),
            "surface_net_solar_radiation": (("time", "basin"), df[['surface_net_solar_radiation']].values),
            "surface_net_thermal_radiation": (("time", "basin"), df[['surface_net_thermal_radiation']].values),
            "surface_pressure": (("time", "basin"), df[['surface_pressure']].values),
            "temperature_2m": (("time", "basin"), df[['temperature_2m']].values),
            "total_precipitation": (("time", "basin"), df[['total_precipitation']].values),
            "u_component_of_wind_10m": (("time", "basin"), df[['u_component_of_wind_10m']].values),
            "v_component_of_wind_10m": (("time", "basin"), df[['v_component_of_wind_10m']].values),
            "volumetric_soil_water_layer_1": (("time", "basin"), df[['volumetric_soil_water_layer_1']].values),
            "volumetric_soil_water_layer_2": (("time", "basin"), df[['volumetric_soil_water_layer_2']].values),
            "volumetric_soil_water_layer_3": (("time", "basin"), df[['volumetric_soil_water_layer_3']].values),
            "volumetric_soil_water_layer_4": (("time", "basin"), df[['volumetric_soil_water_layer_4']].values),
            "geo": (("time", "basin"), df[['geo']].values),
        },
        coords={
            "time": df['TM'].values,
            "basin": [basin_value]
        }
    )
    # 将 'time' 坐标转换为 datetime64[ns] 类型
    ds['time'] = ds['time'].astype('datetime64[ns]')
    # 将数据集保存为 NetCDF 文件
    nc_file_path = 'data/02051500/02051500.nc'  # 替换为您想要保存的路径
    ds.to_netcdf(nc_file_path)

def concat_nc():
    # Load the NetCDF files
    file1 = 'data/02051500/02051500.nc'  # Replace with your file path
    file2 = 'data/21401550/21401550.nc' # Replace with your file path

    # Open the files using xarray
    ds1 = xr.open_dataset(file1)
    ds2 = xr.open_dataset(file2)

    # Merge the datasets along the 'basin' dimension
    combined_ds = xr.concat([ds1, ds2], dim='basin')

    # Save the combined dataset to a new NetCDF file using NETCDF4 format and specifying the NetCDF4 backend
    output_file_path = 'data/combined_dataset.nc'  # Replace with your desired output file path
    combined_ds.to_netcdf(output_file_path)