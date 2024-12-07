import glob
import os
import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

file_list0 = glob.glob('/ftproot/iowa_stations0/*.csv', recursive=True)
file_list1 = glob.glob('/ftproot/iowa_stations1/*.csv', recursive=True)
file_list2 = glob.glob('/ftproot/iowa_stations2/*.csv', recursive=True)
file_list3 = glob.glob('/ftproot/iowa_stations3/*.csv', recursive=True)
file_list4 = glob.glob('/ftproot/iowa_stations4/*.csv', recursive=True)
file_list5 = glob.glob('/ftproot/iowa_stations5/*.csv', recursive=True)
file_list = file_list0 + file_list1 + file_list2 + file_list3 + file_list4 + file_list5


def test_gen_table_heads():
    head_dict = {}
    for file in file_list:
        file_name = file.split('/')[-1].split('.')[0]
        df = pd.read_csv(file, engine='c')
        head_dict[file_name] = df.columns.tolist()
    with open('heads.txt', 'w') as fp:
        for key in head_dict.keys():
            fp.writelines(key+':'+str(head_dict[key]))

def test_convert_iowa_stream_datasets():
    q_list = []
    for file in file_list:
        sta_df = pd.read_csv(file, engine='c')
        sta_columns = sta_df.columns.to_numpy().astype(str)
        if np.any(np.char.startswith(sta_columns, 'Q')):
            q_list.append(file.split('/')[-1])
    print(q_list)
    print(len(q_list))

def test_check_stream_tables():
    stations = pd.read_csv('q_stations.txt')
    true_file_paths = infer_path_from_list(stations['station'].tolist())
    total_df = pd.DataFrame()
    for file_path in true_file_paths:
        df = pd.read_csv(file_path, engine='c')
        if 'key' in df.columns:
            df = df.rename(columns={'Unnamed: 0_level_0': 'station', 'key': 'utc_valid'})
            df = df.drop(0).reset_index()
        df['utc_valid'] = pd.to_datetime(df['utc_valid'])
        df = df.set_index('utc_valid')
        flow_df = df[df.columns[df.columns.str.startswith('Q')]]
        flow_max_count_col = flow_df.count().idxmax()
        flow_df[flow_max_count_col] = flow_df[flow_max_count_col].astype(float)
        flow_df[flow_max_count_col] = flow_df.apply(fill_na_with_other_cols, args=(flow_max_count_col,), axis=1)
        flow_df = flow_df.rename(columns={flow_max_count_col: 'streamflow'})
        flow_df = flow_df.resample('h').mean().dropna()
        flow_df = flow_df.reset_index()[['utc_valid', 'streamflow']]
        name = file_path.split('/')[-1].split('.')[0]
        flow_df['station'] = np.repeat(name, len(flow_df))
        total_df = pd.concat([total_df, flow_df])
    total_df = total_df.set_index(['station', 'utc_valid'])
    total_ds = xr.Dataset.from_dataframe(total_df[~total_df.index.duplicated()])
    total_ds.to_netcdf('/ftproot/iowa_streamflow_stas.nc')

def test_check_iowa_pp_data():
    iowa_pp_node_gdf = gpd.read_file('iowa_all_locs/iowa_pp_stations_day.shp', engine='pyogrio')
    true_file_paths = infer_path_from_shp(iowa_pp_node_gdf)
    total_prcp_df = pd.DataFrame()
    for file in true_file_paths:
        if 'DCP' in file:
            df = pd.read_csv(file, engine='c')
            if 'key' in df.columns:
                df = df.rename(columns={'Unnamed: 0_level_0': 'station', 'key': 'utc_valid'})
                df = df.drop(0).reset_index()
            df['utc_valid'] = pd.to_datetime(df['utc_valid'])
            df = df.set_index('utc_valid')
            prcp_columns = df.columns[df.columns.str.startswith('P')]
            prcp_df = df[prcp_columns]
            pc_max_count_col = prcp_df.count().idxmax()
            prcp_df[pc_max_count_col] = prcp_df[pc_max_count_col].astype(float)
            if pc_max_count_col.startswith('PC'):
                prcp_df = prcp_df[prcp_df.columns[prcp_df.columns.str.startswith('PC')]]
                prcp_df[pc_max_count_col] = prcp_df.apply(fill_na_with_other_cols, args=(pc_max_count_col,), axis=1)
                prcp_df['year'] = df.index.year
                prcp_df = prcp_df.groupby('year').apply(calculate_differences, pc_max_count_col).drop('year', axis=1)
                prcp_df = prcp_df.reset_index().drop(columns=['year']).set_index('utc_valid')
            elif pc_max_count_col.startswith('PP') or pc_max_count_col.startswith('PR'):
                prcp_df = prcp_df[prcp_df.columns[prcp_df.columns.str.startswith('PP') | prcp_df.columns.str.startswith('PR')]]
                prcp_df[pc_max_count_col] = prcp_df.apply(fill_na_with_other_cols, args=(pc_max_count_col,), axis=1)
                prcp_df = prcp_df.rename(columns={pc_max_count_col: 'prcp_inch'})
            else:
                continue
            prcp_arr = prcp_df['prcp_inch'].astype(float).to_numpy()
            minus_index = np.argwhere(prcp_arr<0)
            oppo_index = np.add(minus_index, 1)
            error_index = np.append(minus_index, oppo_index)
            if len(prcp_arr) in error_index:
                error_index = np.delete(error_index, np.argwhere(error_index==len(prcp_arr)))
            prcp_arr[error_index] = 0
            prcp_arr[prcp_arr>10] = 0
            prcp_df['prcp_inch'] = prcp_arr
            prcp_df = pd.DataFrame(prcp_df['prcp_inch'])
        elif 'ASOS' in file:
            df = pd.read_csv(file, engine='c')
            df['valid'] = pd.to_datetime(df['valid'])
            df = df.set_index('valid')
            prcp_df = pd.DataFrame(df['p01i']).rename(columns={'p01i':'prcp_inch'})
        else:
            continue
        name = file.split('_')[-1].split('.')[0]
        prcp_df = prcp_df.resample('h').sum().dropna().reset_index()
        prcp_df['station_id'] = np.repeat(name, len(prcp_df))
        total_prcp_df = pd.concat([total_prcp_df, prcp_df])
    total_prcp_df = total_prcp_df.set_index(['station_id', 'utc_valid'])
    res_ds = xr.Dataset.from_dataframe(total_prcp_df.dropna())
    res_ds = res_ds.to_netcdf('/ftproot/iowa_prcp_data_day.nc')
    return res_ds

def test_read_iowa_by_camels():
    # 检查NOAA站点的数据；
    # 使用iowa的1833个站点（放弃时间要求）
    node_shp_gdf = gpd.read_file('iowa_all_locs/_ALL__locs.shp', engine='pyogrio')
    basin_shp_gdf = gpd.read_file('iowa_all_locs/basins_shp.shp', engine='pyogrio')
    nodes_intersect = gpd.sjoin(node_shp_gdf, basin_shp_gdf)
    local_nodes = nodes_intersect[nodes_intersect['NETWORK'].str.contains('|'.join(['DCP', 'COOP', 'ASOS']))]
    true_file_paths = infer_path_from_shp(local_nodes)
    # 先统计一下数据缺失率(按小时或按天)
    # 不跳空是因为站点可能会换变量统计，跳空无意义
    loss_dict = {}
    start_times = []
    end_times = []
    for file in np.unique(true_file_paths):
        name = file.split('/')[-1].split('.')[0]
        df = pd.read_csv(file, engine='c')
        if 'DCP' in name:
            if 'key' in df.columns:
                df = df.rename(columns={'Unnamed: 0_level_0': 'station', 'key': 'utc_valid'})
                df = df.drop(0).reset_index()
            df['utc_valid'] = pd.to_datetime(df['utc_valid'])
            start_times.append(df['utc_valid'][0])
            end_times.append(df['utc_valid'].iloc[-1])
            df = df.set_index('utc_valid')
        elif 'COOP' in name:
            df['valid_time'] = pd.to_datetime(df['date']+' '+df['time'])
            df = df[~df['valid_time'].isna()].reset_index()
            if len(df)!=0:
                start_times.append(df['valid_time'][0])
                end_times.append(df['valid_time'].iloc[-1])
                df = df.set_index('valid_time')
            else:
                start_times.append(np.nan)
                end_times.append(np.nan)
                loss_dict[name]=1
                continue
        elif 'ASOS' in name:
            df['valid'] = pd.to_datetime(df['valid'])
            start_times.append(df['valid'][0])
            end_times.append(df['valid'].iloc[-1])
            df = df.set_index('valid')
        else:
            loss_dict[name]=1
            continue
        resample_df = df.resample('d').last()
        loss_dict[name] = max(0.0, 1 - len(df) / len(resample_df))
    res_df = pd.DataFrame.from_dict(loss_dict, orient='index').reset_index().rename(columns={'index':'name', 0: 'loss_rate'})
    res_df['start_time'] = start_times
    res_df['end_time'] = end_times
    res_df.to_csv('iowa_time_loss_rate_day.csv')

def test_gen_iowa_stream_gdf():
    sta_all = gpd.read_file('iowa_all_locs/_ALL__locs.shp')
    stations_ds = xr.open_dataset('/ftproot/iowa_streamflow_stas.nc')
    stations = stations_ds['station'].values
    iowa_ids = [sta.split('_')[-1] for sta in stations]
    stream_stations = sta_all[sta_all['ID'].isin(iowa_ids)]
    stream_stations['ID'] = stream_stations['NETWORK'] + '_' + stream_stations['ID']
    stream_stations.to_file('iowa_all_locs/iowa_stream_stations.shp')

def test_analyze_iowa_loss():
    # 23/24 = 0.958, 故如果按日取数，缺失率将在0.96左右，再高就是连日尺度都缺失严重
    loss_df = pd.read_csv('iowa_time_loss_rate_day.csv', engine='c')
    better_names = loss_df['name'][loss_df['loss_rate']<0.5]
    ids = better_names.apply(lambda x: x.split('_')[-1])
    node_shp_gdf = gpd.read_file('iowa_all_locs/_ALL__locs.shp', engine='pyogrio')
    inter_node_shp_gdf = node_shp_gdf[node_shp_gdf['ID'].isin(ids)]
    true_file_paths = infer_path_from_shp(inter_node_shp_gdf)
    pp_stas = []
    for file in true_file_paths:
        df = pd.read_csv(file, engine='c')
        prcp_columns = df.columns[df.columns.str.startswith('P')]
        if len(prcp_columns) > 0:
            pp_stas.append(file.split('_')[-1].split('.')[0])
    pp_inter_node_shp_gdf = inter_node_shp_gdf[inter_node_shp_gdf['ID'].isin(pp_stas)]
    pp_inter_node_shp_gdf.to_file('iowa_all_locs/iowa_pp_stations_day.shp')

def infer_path_from_list(local_file_names):
    true_file_paths = []
    for file_name in local_file_names:
        for num in range(6):
            if os.path.exists(f"/ftproot/iowa_stations{num}/{file_name}"):
                file_path = f"/ftproot/iowa_stations{num}/{file_name}"
                true_file_paths.append(file_path)
                break
    return true_file_paths

def infer_path_from_shp(iowa_node_gdf):
    local_file_names = (iowa_node_gdf['NETWORK'] + '_' + iowa_node_gdf['ID']).to_list()
    true_file_paths = []
    for file_name in local_file_names:
        for num in range(6):
            if os.path.exists(f"/ftproot/iowa_stations{num}/{file_name}.csv"):
                file_path = f"/ftproot/iowa_stations{num}/{file_name}.csv"
                true_file_paths.append(file_path)
                break
    return true_file_paths

def fill_na_with_other_cols(row, col_name):
    # 获取基准列的值
    value = row[col_name]
    if pd.isna(value):
        # 如果基准列的值为空，则尝试从其他列获取值
        for col in row.index:
            if col != col_name and not pd.isna(row[col]):
                return row[col]
    return value

def calculate_differences(group, max_col_name):
    group['prcp_inch'] = group[max_col_name].astype(float).diff().fillna(0)
    return group

def replace_zeros_with_mode(x):
    mode_value = x.mode().iloc[0] if len(x.mode()) > 0 else 0
    x[x == 0] = mode_value
    return x
