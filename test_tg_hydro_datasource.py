#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试TgHydroDatasource类的功能
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from shapely.geometry import Polygon
from pathlib import Path

# 添加项目路径到sys.path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from hydrodatasource.reader.data_source import TgHydroDatasource


def create_test_data_structure(test_data_dir):
    """创建测试数据目录结构"""
    print("创建测试数据目录结构...")
    
    # 创建目录结构
    dirs_to_create = [
        "attributes",
        "timeseries/1D",
        "shapes", 
        "tggnn"
    ]
    
    for dir_name in dirs_to_create:
        dir_path = os.path.join(test_data_dir, dir_name)
        os.makedirs(dir_path, exist_ok=True)
        print(f"创建目录: {dir_path}")


def create_test_graph_dict(test_data_dir):
    """创建测试用的graph_dict.json文件"""
    print("创建测试graph_dict.json文件...")
    
    graph_dict = {
        'test_basin': [
            ['station_1', 'station_2', 'station_3'],
            ['station_4', 'station_2', 'station_3'],
            ['station_5', 'station_6', 'station_3']
        ]
    }
    
    graph_dict_file = os.path.join(test_data_dir, "tggnn", "graph_dict.json")
    with open(graph_dict_file, 'w', encoding='utf-8') as f:
        json.dump(graph_dict, f, ensure_ascii=False, indent=2)
    
    print(f"创建文件: {graph_dict_file}")
    return graph_dict


def create_test_lstm_predictions(test_data_dir):
    """创建测试用的lstmpred.nc文件"""
    print("创建测试lstmpred.nc文件...")
    
    # 创建测试数据
    time_range = pd.date_range('2020-01-01', '2020-01-10', freq='H')
    basin_names = ['station_1', 'station_2', 'station_3', 'station_4', 'station_5', 'station_6']
    
    # 生成随机流量数据
    np.random.seed(42)
    streamflow_data = np.random.rand(len(basin_names), len(time_range)) * 100
    
    # 创建xarray Dataset
    lstm_dataset = xr.Dataset({
        'streamflow': (['basin', 'time'], streamflow_data)
    }, coords={
        'basin': basin_names,
        'time': time_range
    })
    
    lstm_pred_file = os.path.join(test_data_dir, "tggnn", "lstmpred.nc")
    lstm_dataset.to_netcdf(lstm_pred_file)
    
    print(f"创建文件: {lstm_pred_file}")
    return lstm_dataset


def create_test_timeseries(test_data_dir):
    """创建测试用的时间序列数据"""
    print("创建测试时间序列数据...")
    
    # 创建测试数据
    time_range = pd.date_range('2020-01-01', '2020-01-10', freq='H')
    basin_names = ['station_1', 'station_2', 'station_3', 'station_4', 'station_5', 'station_6']
    
    # 生成随机数据
    np.random.seed(42)
    precipitation_data = np.random.rand(len(basin_names), len(time_range)) * 10
    streamflow_data = np.random.rand(len(basin_names), len(time_range)) * 50
    
    # 创建xarray Dataset
    timeseries_dataset = xr.Dataset({
        'total_precipitation_hourly': (['basin', 'time'], precipitation_data),
        'streamflow': (['basin', 'time'], streamflow_data)
    }, coords={
        'basin': basin_names,
        'time': time_range
    })
    
    timeseries_file = os.path.join(test_data_dir, "timeseries", "1D", "timeseries_1D.nc")
    timeseries_dataset.to_netcdf(timeseries_file)
    
    print(f"创建文件: {timeseries_file}")
    return timeseries_dataset


def create_test_shapes(test_data_dir):
    """创建测试用的形状文件"""
    print("创建测试形状文件...")
    
    basin_names = ['station_1', 'station_2', 'station_3', 'station_4', 'station_5', 'station_6']
    
    # 创建简单的多边形几何
    geometries = []
    for i, basin_name in enumerate(basin_names):
        # 创建简单的矩形多边形 (使用经纬度坐标)
        x_offset = 100 + i * 0.1  # 经度
        y_offset = 30 + i * 0.1   # 纬度
        polygon = Polygon([
            (x_offset, y_offset),
            (x_offset + 0.05, y_offset),
            (x_offset + 0.05, y_offset + 0.05),
            (x_offset, y_offset + 0.05)
        ])
        geometries.append(polygon)
    
    # 创建GeoDataFrame并设置CRS为WGS84
    gdf = gpd.GeoDataFrame({
        'BASIN_ID': basin_names,  # 使用BASIN_ID列名
        'geometry': geometries
    }, crs='EPSG:4326')  # WGS84坐标系
    
    shapes_file = os.path.join(test_data_dir, "shapes", "basins.shp")
    gdf.to_file(shapes_file)
    
    print(f"创建文件: {shapes_file}")
    return gdf


def create_test_unit_info(test_data_dir):
    """创建测试用的单位信息文件"""
    print("创建测试单位信息文件...")
    
    # 创建单位信息字典
    unit_info = {
        "ele_mt_sav": "m",
        "area": "km2", 
        "p_mean": "mm",
        "total_precipitation_hourly": "mm/h",
        "streamflow": "mm/d"
    }
    
    unit_file = os.path.join(test_data_dir, "timeseries", "1D_units_info.json")
    with open(unit_file, 'w', encoding='utf-8') as f:
        json.dump(unit_info, f, ensure_ascii=False, indent=2)
    
    print(f"创建文件: {unit_file}")
    return unit_info


def create_test_attributes(test_data_dir):
    """创建测试用的属性数据"""
    print("创建测试属性数据...")
    
    basin_names = ['station_1', 'station_2', 'station_3', 'station_4', 'station_5', 'station_6']
    
    # 生成随机属性数据
    np.random.seed(42)
    ele_mt_sav = np.random.rand(len(basin_names)) * 1000 + 500  # 海拔
    area = np.random.rand(len(basin_names)) * 5000 + 1000      # 面积
    p_mean = np.random.rand(len(basin_names)) * 2000 + 800     # 平均降水
    
    # 创建DataFrame并保存为CSV
    attributes_df = pd.DataFrame({
        'basin_id': basin_names,
        'ele_mt_sav': ele_mt_sav,
        'area': area,
        'p_mean': p_mean
    })
    
    attributes_file = os.path.join(test_data_dir, "attributes", "attributes.csv")
    attributes_df.to_csv(attributes_file, index=False)
    
    print(f"创建文件: {attributes_file}")
    return attributes_df


def test_tg_hydro_datasource():
    """测试TgHydroDatasource类"""
    print("=" * 60)
    print("开始测试TgHydroDatasource类")
    print("=" * 60)
    
    # 设置测试数据目录
    test_data_dir = "/tmp/test_tg_hydro_data"
    
    try:
        # 1. 创建测试数据
        create_test_data_structure(test_data_dir)
        graph_dict = create_test_graph_dict(test_data_dir)
        lstm_data = create_test_lstm_predictions(test_data_dir)
        timeseries_data = create_test_timeseries(test_data_dir)
        shapes_data = create_test_shapes(test_data_dir)
        unit_info = create_test_unit_info(test_data_dir)
        attributes_data = create_test_attributes(test_data_dir)
        
        # 2. 初始化TgHydroDatasource
        print("\n" + "=" * 40)
        print("测试TgHydroDatasource初始化")
        print("=" * 40)
        
        tg_datasource = TgHydroDatasource(
            data_path=test_data_dir,
            time_unit=['1D'],
            dataset_name="test_tg_dataset"
        )
        
        print(f"✓ TgHydroDatasource初始化成功")
        print(f"  数据源名称: {tg_datasource.get_name()}")
        print(f"  数据路径: {tg_datasource.data_source_dir}")
        
        # 3. 测试数据源描述
        print("\n" + "=" * 40)
        print("测试数据源描述")
        print("=" * 40)
        
        data_desc = tg_datasource.data_source_description
        print("数据源描述包含的路径:")
        for key, value in data_desc.items():
            if 'TGGNN' in key or 'LSTM' in key or 'GRAPH' in key:
                print(f"  {key}: {value}")
        
        # 4. 测试图网络结构加载
        print("\n" + "=" * 40)
        print("测试图网络结构")
        print("=" * 40)
        
        print(f"✓ 图网络结构加载成功")
        print(f"  包含流域: {list(tg_datasource.graph_dict.keys())}")
        
        # 5. 测试LSTM预测数据读取
        print("\n" + "=" * 40)
        print("测试LSTM预测数据读取")
        print("=" * 40)

        # 获取实际可用的basin ID
        available_basins = tg_datasource.read_object_ids()
        test_basins = available_basins[:3]  # 使用前三个basin

        lstm_predictions = tg_datasource.read_lstm_predictions(
            object_ids=test_basins[:2],
            t_range_list=['2020-01-01', '2020-01-05']
        )
        
        # 5. 测试时间序列数据加载
        print("\n5. 测试时间序列数据加载...")
        ts_data, lstm_data = tg_datasource.load_ts_data(
            basin_names=test_basins[:2],
            t_range=['2020-01-01', '2020-01-05'],
            var_lst=['total_precipitation_hourly', 'streamflow']
        )
        print(f"✓ LSTM预测数据读取成功")
        print(f"  数据形状: {lstm_predictions['streamflow'].shape}")
        print(f"  时间范围: {lstm_predictions.time.values[0]} 到 {lstm_predictions.time.values[-1]}")
        print(f"✓ 时间序列数据加载成功")
        print(f"  时间序列数据形状: {ts_data.shape}")
        print(f"  LSTM数据形状: {lstm_data.shape}")
        
        # 6. 测试节点属性加载
        print("\n" + "=" * 40)
        print("测试节点属性加载")
        print("=" * 40)
        node_attrs = tg_datasource.load_node_attributes(
            basin_names=test_basins[:3],
            selected_attrs=['ele_mt_sav', 'area', 'p_mean']
        )
        print(f"✓ 节点属性加载成功")
        print(f"  属性张量形状: {node_attrs.shape}")
        
        # 7. 测试图生成
        print("\n" + "=" * 40)
        print("测试图生成")
        print("=" * 40)
        
        basin_names_all = ['station_1', 'station_2', 'station_3', 'station_4', 'station_5', 'station_6']
        dg, edges, node_mapping = tg_datasource.gen_nx_graph('test_basin', basin_names_all)
        print(f"✓ 图生成成功")
        print(f"  节点数量: {len(dg.nodes)}")
        print(f"  边数量: {len(edges)}")
        print(f"  节点映射: {node_mapping}")
        
        # 8. 测试上游节点查找
        print("\n" + "=" * 40)
        print("测试上游节点查找")
        print("=" * 40)
        
        upstream_nodes = tg_datasource.get_upstream_nodes('test_basin', 'station_3', basin_names_all)
        print(f"✓ 上游节点查找成功")
        print(f"  station_3的上游节点索引: {upstream_nodes}")
        
        print("\n" + "=" * 60)
        print("所有测试通过！TgHydroDatasource类工作正常")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # 清理测试数据
        import shutil
        if os.path.exists(test_data_dir):
            shutil.rmtree(test_data_dir)
            print(f"\n清理测试数据目录: {test_data_dir}")


if __name__ == "__main__":
    success = test_tg_hydro_datasource()
    sys.exit(0 if success else 1)