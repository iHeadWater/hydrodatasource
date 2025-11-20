#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化后的TgHydroDatasource类使用示例

该示例展示了如何使用简化后的TgHydroDatasource类进行数据读取，
不涉及torch转换，保持数据的原始xarray格式。
"""

import os
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from hydrodatasource.reader.data_source import TgHydroDatasource


def example_usage():
    """展示TgHydroDatasource的基本使用方法"""
    
    # 假设数据路径
    data_path = "/path/to/your/tg_hydro_data"
    
    # 初始化TgHydroDatasource
    tg_datasource = TgHydroDatasource(
        data_path=data_path,
        time_unit=['1D'],
        dataset_name="example_dataset"
    )
    
    print(f"数据源名称: {tg_datasource.get_name()}")
    print(f"数据路径: {tg_datasource.data_source_dir}")
    
    # 1. 读取可用的流域ID
    available_basins = tg_datasource.read_object_ids()
    print(f"可用流域数量: {len(available_basins)}")
    print(f"前5个流域ID: {available_basins[:5]}")
    
    # 2. 读取时间序列数据和LSTM预测数据
    selected_basins = available_basins[:3]  # 选择前3个流域
    time_range = ['2020-01-01', '2020-12-31']
    variables = ['total_precipitation_hourly', 'streamflow']
    
    timeseries_data, lstm_data = tg_datasource.read_timeseries_with_lstm(
        object_ids=selected_basins,
        t_range=time_range,
        var_lst=variables
    )
    
    print(f"\n时间序列数据:")
    print(f"  数据类型: {type(timeseries_data)}")
    print(f"  时间单位: {list(timeseries_data.keys())}")
    for time_unit, data in timeseries_data.items():
        print(f"  {time_unit} 数据形状: {data.dims}")
        print(f"  {time_unit} 变量: {list(data.data_vars)}")
    
    print(f"\nLSTM预测数据:")
    print(f"  数据类型: {type(lstm_data)}")
    print(f"  数据形状: {lstm_data.dims}")
    print(f"  变量: {list(lstm_data.data_vars)}")
    
    # 3. 读取属性数据
    selected_attrs = ['ele_mt_sav', 'area', 'p_mean']
    attr_data = tg_datasource.read_node_attributes(
        object_ids=selected_basins,
        selected_attrs=selected_attrs
    )
    
    print(f"\n属性数据:")
    print(f"  数据类型: {type(attr_data)}")
    print(f"  数据形状: {attr_data.dims}")
    print(f"  变量: {list(attr_data.data_vars)}")
    
    # 4. 生成图网络结构
    basin_id = "test_basin"  # 根据实际情况修改
    try:
        dg, edges, node_mapping, valid_object_ids = tg_datasource.gen_nx_graph(
            basin_id=basin_id,
            object_ids=selected_basins
        )
        
        print(f"\n图网络结构:")
        print(f"  节点数量: {len(dg.nodes)}")
        print(f"  边数量: {len(edges)}")
        print(f"  节点映射: {node_mapping}")
        print(f"  有效对象ID: {valid_object_ids}")
        
        # 5. 获取上游节点
        if valid_object_ids:
            target_node = valid_object_ids[0]
            upstream_nodes = tg_datasource.get_upstream_nodes(
                basin_id=basin_id,
                target_node=target_node,
                object_ids=selected_basins
            )
            print(f"  {target_node} 的上游节点索引: {upstream_nodes}")
            
    except ValueError as e:
        print(f"\n图网络结构错误: {e}")
        print("请检查graph_dict.json文件是否存在且包含正确的流域ID")
    
    # 6. 使用父类方法读取数据（展示继承功能）
    print(f"\n使用父类方法读取数据:")
    
    # 读取时间序列数据
    ts_data = tg_datasource.read_ts_xrdataset(
        gage_id_lst=selected_basins,
        t_range=time_range,
        var_lst=['streamflow'],
        time_units=['1D']
    )
    print(f"  时间序列数据 (父类方法): {type(ts_data)}")
    
    # 读取属性数据
    attr_data_parent = tg_datasource.read_attr_xrdataset(
        var_lst=['area']
    )
    print(f"  属性数据 (父类方法): {type(attr_data_parent)}")


def data_processing_example():
    """展示如何处理读取的数据"""
    
    print("\n" + "="*50)
    print("数据处理示例")
    print("="*50)
    
    # 这里展示如何处理从TgHydroDatasource读取的xarray数据
    # 用户可以根据需要进行各种数据处理和分析
    
    print("1. 数据筛选和切片")
    print("   - 使用 .sel() 方法按坐标筛选")
    print("   - 使用 .isel() 方法按索引筛选")
    print("   - 使用 .where() 方法按条件筛选")
    
    print("\n2. 数据转换")
    print("   - 使用 .values 获取numpy数组")
    print("   - 使用 .to_pandas() 转换为pandas")
    print("   - 使用 .to_netcdf() 保存为NetCDF文件")
    
    print("\n3. 数据分析")
    print("   - 使用 .mean(), .std(), .max(), .min() 等统计方法")
    print("   - 使用 .groupby() 进行分组分析")
    print("   - 使用 .resample() 进行时间重采样")
    
    print("\n4. 如果需要torch张量，可以手动转换:")
    print("   import torch")
    print("   tensor = torch.from_numpy(data.values).float()")


if __name__ == "__main__":
    print("TgHydroDatasource 简化版使用示例")
    print("="*50)
    
    try:
        example_usage()
        data_processing_example()
        
        print("\n" + "="*50)
        print("示例完成！")
        print("="*50)
        
    except Exception as e:
        print(f"运行示例时出错: {e}")
        print("请确保数据路径正确且数据文件存在")