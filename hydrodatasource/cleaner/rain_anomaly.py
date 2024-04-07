import xarray as xr
import pandas as pd
import numpy as np
import os

# 对比era5数据

# 数据极值筛查

# 空间数据对比

# 数据格式规范
def rainfall_format_normalization(df,freq='h'):
    # 转换时间列为datetime类型
    df['TM'] = pd.to_datetime(df['TM'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    # 尝试转换STCD列为整数，如果失败，则转换为字符串
    try:
        df['STCD'] = df['STCD'].astype(int).astype(str)
    except ValueError:
        df['STCD'] = df['STCD'].astype(str)
    
    # 生成完整的时间序列，确保没有间隔
    full_time_series = pd.date_range(start=df['TM'].min(), end=df['TM'].max(), freq='h')
    full_df = pd.DataFrame(full_time_series, columns=['TM'])
    
    # 确保合并前两个DataFrame的时间列数据类型一致
    df['TM'] = pd.to_datetime(df['TM'])
    full_df['TM'] = pd.to_datetime(full_df['TM'])
    
    # 合并原始数据到完整的时间序列中
    df_complete = pd.merge(full_df, df, on='TM', how='left')
    
    # 插补缺失数据
    stcd_fill_value = df['STCD'].dropna().iloc[0] if not df['STCD'].dropna().empty else None
    df_complete['DRP'].fillna(0, inplace=True)
    df_complete['STCD'].fillna(stcd_fill_value, inplace=True)

    return df_complete