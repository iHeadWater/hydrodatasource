import polars as pl
import numpy as np
import os
from tqdm.asyncio import tqdm

def find_continuous_intervals_vectorized(arr):
    """
    找出数组中所有公差为 1 的连续区间（向量化实现）。
    :param arr: 单调递增的 numpy 数组
    :return: 列表，每个元素是一个公差为 1 的区间（子数组）
    """
    if len(arr) == 0:
        return []
    diffs = np.diff(arr)
    boundaries = np.where(diffs != 1)[0]
    boundaries = np.concatenate(([-1], boundaries, [len(arr) - 1]))
    intervals = np.split(arr, boundaries + 1)
    intervals = [interval for interval in intervals if len(interval) > 1]
    return intervals


def adaptive_ewma_with_ewm(streamflow_data, alpha_high=0.6, alpha_low=0.3):
    """
    自适应EWMA方法：基于数据的统计特性动态调整alpha值，并使用 pandas ewm() 进行计算。
    :param streamflow_data: 流量数据
    :param alpha_high: 流量波动较大时的alpha值（较平滑）
    :param alpha_low: 流量波动较小时的alpha值（较不平滑）
    :return: 自适应平滑后的数据
    """
    streamflow_series = pl.Series(streamflow_data)
    Q1 = streamflow_series.quantile(0.25)
    Q3 = streamflow_series.quantile(0.75)
    IQR = Q3 - Q1
    # 使用IQR和标准差来动态调整阈值
    dynamic_threshold = Q3 + 1.5 * IQR  # 常见的阈值规则
    adaptive_alphas = np.where(streamflow_series >= dynamic_threshold, alpha_high, alpha_low)
    ewma_data = np.zeros_like(streamflow_data)
    ewma_high = streamflow_series.ewm_mean(alpha=alpha_high)
    ewma_low = streamflow_series.ewm_mean(alpha=alpha_low)
    ewma_high_index = np.argwhere(adaptive_alphas == alpha_high)
    ewma_low_index = np.argwhere(adaptive_alphas == alpha_low)
    alpha_zones_high = find_continuous_intervals_vectorized(np.concatenate(ewma_high_index)) if len(ewma_high_index>0) else []
    alpha_zones_low = find_continuous_intervals_vectorized(np.concatenate(ewma_low_index)) if len(ewma_low_index>0) else []
    alpha_zones = alpha_zones_high + alpha_zones_low
    # np.array_equal(streamflow_series.ewm_mean(alpha=current_alpha)[:i+1].to_numpy(), streamflow_series[:i+1].ewm_mean(alpha=current_alpha).to_numpy()) = True
    for a_zone in alpha_zones:
        ewma_all = ewma_high if adaptive_alphas[a_zone[0]] == alpha_high else ewma_low
        ewma_data[a_zone[0]: a_zone[-1]+1] = ewma_all[a_zone[0]: a_zone[-1]+1]
    return ewma_data

def process_csv_with_dynamic_adaptive_ewma(file_name, alpha_high=0.6, alpha_low=0.3):
    """
    处理CSV文件，应用自适应EWMA方法，自动调整阈值。
    :param file_name: 原csv文件路径
    :param alpha_high: 流量波动较大时的alpha值
    :param alpha_low: 流量波动较小时的alpha值
    """
    csv_path = os.path.join(os.getcwd(), 'tests/flowdb_data', file_name.split('/')[-1]+'_fixed.csv')
    if os.path.exists(csv_path):
        print(f"跳过: {file_name}, 文件已存在")
    else:
        df = pl.read_csv(file_name + '.csv', schema_overrides={'00060': pl.Float64})
        if '00060' in df.columns:
            df = df.with_row_index('row_idx')
            # 记录原始缺失值的位置(部分数据为-999999，也需要滤除)
            missing_mask = df.filter((pl.col('00060').is_null()) | (pl.col('00060').is_nan()) | (pl.col('00060')<=-100))
            missing_index = missing_mask['row_idx'].to_numpy()
            df = df.with_columns(pl.when(pl.col('row_idx').is_in(missing_index)).then(0).otherwise(pl.col('00060')).alias('00060'))
            streamflow_data = df['00060'].to_numpy()
            ewma_transformed_data = adaptive_ewma_with_ewm(streamflow_data, alpha_high, alpha_low)
            df = df.with_columns(pl.Series(data_balanced(streamflow_data, ewma_transformed_data)).alias('AEWMA'))
            # 将原本缺失值的位置重新设置为 NaN
            df = df.with_columns(pl.when(pl.col('row_idx').is_in(missing_index)).then(np.nan).otherwise(pl.col('00060')).alias('00060'))
            df = df.with_columns(pl.when(pl.col('row_idx').is_in(missing_index)).then(np.nan).otherwise(pl.col('AEWMA')).alias('AEWMA'))
            df.write_csv(csv_path)
            print(f"处理完成: {file_name}")
        else:
            print(f"跳过: {file_name}, 未找到 '00060' 列")

def data_balanced(origin_data, transform_data):
    """
    对一维流量数据进行总量平衡变换。
    :origin_data: 原始一维流量数据。
    :transform_data: 平滑转换后的一维流量数据。
    """
    # Calculate the flow balance factor and keep the total volume consistent
    streamflow_data_before = np.sum(origin_data)
    streamflow_data_after = np.sum(transform_data)
    scaling_factor = streamflow_data_before / streamflow_data_after
    balanced_data = transform_data * scaling_factor
    print(f"Total flow (before smoothing): {streamflow_data_before}")
    print(f"Total flow (after smoothing): {np.sum(balanced_data)}")
    return balanced_data

# 示例调用
if __name__ == '__main__':
    import geopandas as gpd
    stations_shp = gpd.read_file("/ftproot/flowdb_locs/flowdb_stations.shp", engine='pyogrio')
    stations_list = stations_shp['BASIN_ID'].tolist()
    folder_path = "/ftproot/usgs_camels_hourly_flowdb_1373/"
    for station in tqdm(stations_list):
        file_name = os.path.join(folder_path, f'zq_USA_usgs_{station}')
        process_csv_with_dynamic_adaptive_ewma(file_name)
