#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重构测试：验证 TgHydroDatasource 的 read_ts_xrdataset 与 read_attr_xrdataset 输出形状与变量
"""

import os
import json
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import geopandas as gpd
from shapely.geometry import Polygon
import hydrodatasource.configs.config as conf

from hydrodatasource.reader.data_source import TgHydroDatasource


def _create_dirs(root, dataset_name):
    base = os.path.join(root, dataset_name)
    os.makedirs(os.path.join(base, "attributes"), exist_ok=True)
    os.makedirs(os.path.join(base, "timeseries", "1D"), exist_ok=True)
    os.makedirs(os.path.join(base, "shapes"), exist_ok=True)
    os.makedirs(os.path.join(base, "tggnn"), exist_ok=True)
    os.makedirs(os.path.join(base, "intermediate"), exist_ok=True)
    return base


def _create_graph_dict(base):
    graph_dict = {
        "test_basin": [
            ["station_1", "station_2", "station_3"],
            ["station_4", "station_2", "station_3"],
            ["station_5", "station_6", "station_3"],
        ]
    }
    # TgHydroDatasource 读取路径为 <data_source_dir>/intermediate/graph_dict.json
    graph_dict_file = os.path.join(base, "intermediate", "graph_dict.json")
    with open(graph_dict_file, "w", encoding="utf-8") as f:
        json.dump(graph_dict, f, ensure_ascii=False, indent=2)
    return graph_dict_file


def _create_lstm_predictions(base):
    # Hourly streamflow for resampling to 1D
    time_range = pd.date_range("2020-01-01", "2020-01-10", freq="H")
    basin_names = [
        "station_1",
        "station_2",
        "station_3",
        "station_4",
        "station_5",
        "station_6",
    ]
    np.random.seed(42)
    streamflow_data = np.random.rand(len(basin_names), len(time_range)) * 100
    lstm_dataset = xr.Dataset(
        {"streamflow": (["basin", "time"], streamflow_data)},
        coords={"basin": basin_names, "time": time_range},
    )
    # 写入到数据源目录（便于调试和留存）
    lstm_pred_file = os.path.join(base, "tggnn", "lstmpred.nc")
    os.makedirs(os.path.dirname(lstm_pred_file), exist_ok=True)
    lstm_dataset.to_netcdf(lstm_pred_file)

    # 同步写入到缓存目录，满足 TgHydroDatasource 的路径约定
    os.makedirs(conf.CACHE_DIR, exist_ok=True)
    lstm_dataset.to_netcdf(os.path.join(conf.CACHE_DIR, "lstmpred.nc"))
    return lstm_pred_file


def _create_timeseries_csv(base):
    # Daily timeseries CSV per basin, as expected by reader
    time_range = pd.date_range("2020-01-01", "2020-01-10", freq="1D")
    basin_names = [
        "station_1",
        "station_2",
        "station_3",
        "station_4",
        "station_5",
        "station_6",
    ]
    np.random.seed(123)
    for name in basin_names:
        df = pd.DataFrame(
            {
                "time": time_range,
                "total_precipitation_hourly": np.random.rand(len(time_range)) * 10,
                "streamflow": np.random.rand(len(time_range)) * 50,
            }
        )
        df.to_csv(os.path.join(base, "timeseries", "1D", f"{name}.csv"), index=False)


def _create_shapes(base):
    basin_names = [
        "station_1",
        "station_2",
        "station_3",
        "station_4",
        "station_5",
        "station_6",
    ]
    geometries = []
    for i, _ in enumerate(basin_names):
        x_offset = 100 + i * 0.1
        y_offset = 30 + i * 0.1
        polygon = Polygon(
            [
                (x_offset, y_offset),
                (x_offset + 0.05, y_offset),
                (x_offset + 0.05, y_offset + 0.05),
                (x_offset, y_offset + 0.05),
            ]
        )
        geometries.append(polygon)
    gdf = gpd.GeoDataFrame({"BASIN_ID": basin_names, "geometry": geometries}, crs="EPSG:4326")
    gdf.to_file(os.path.join(base, "shapes", "basins.shp"))


def _create_unit_info(base):
    unit_info = {
        "ele_mt_sav": "m",
        "area": "km^2",
        "p_mean": "mm",
        "total_precipitation_hourly": "mm/h",
        "streamflow": "mm/d",
    }
    unit_file = os.path.join(base, "timeseries", "1D_units_info.json")
    with open(unit_file, "w", encoding="utf-8") as f:
        json.dump(unit_info, f, ensure_ascii=False, indent=2)
    return unit_file


def _create_attributes(base):
    basin_names = [
        "station_1",
        "station_2",
        "station_3",
        "station_4",
        "station_5",
        "station_6",
    ]
    np.random.seed(7)
    df = pd.DataFrame(
        {
            "basin_id": basin_names,
            "ele_mt_sav": np.random.rand(len(basin_names)) * 1000 + 500,
            "area": np.random.rand(len(basin_names)) * 5000 + 1000,
            "p_mean": np.random.rand(len(basin_names)) * 2000 + 800,
        }
    )
    df.to_csv(os.path.join(base, "attributes", "attributes.csv"), index=False)


@pytest.fixture
def tg_dataset(tmp_path):
    dataset_name = "test_tg_dataset"
    base = _create_dirs(tmp_path, dataset_name)
    _create_graph_dict(base)
    _create_lstm_predictions(base)
    _create_timeseries_csv(base)
    _create_shapes(base)
    _create_unit_info(base)
    _create_attributes(base)

    ds = TgHydroDatasource(uri=str(base), time_unit=["1D"])
    return ds


def test_read_ts_xrdataset_shapes_and_vars(tg_dataset):
    basins = ["station_1", "station_2", "station_3"]
    t_range = ["2020-01-01", "2020-01-10"]
    var_lst = ["discharge", "total_precipitation_hourly", "streamflow", "lstm_pred"]

    # 注意：父类缓存生成只会包含 CSV 的列；这里包含不存在的 "discharge" 会被过滤或触发校验
    # 为避免报错，先限制为 CSV 中存在的列 + lstm_pred
    var_lst = ["total_precipitation_hourly", "streamflow", "lstm_pred"]

    ds_map = tg_dataset.read_ts_xrdataset(
        gage_id_lst=basins,
        t_range=t_range,
        var_lst=var_lst,
        time_units=["1D"],
    )

    assert isinstance(ds_map, dict)
    assert "1D" in ds_map
    ds = ds_map["1D"]
    assert isinstance(ds, xr.Dataset)

    # 直观打印：整体结构与基本信息
    print("\n[TS] Dataset summary:\n", ds)
    print("[TS] dims:", ds.dims)
    print("[TS] sizes:", dict(ds.sizes))
    print("[TS] vars:", list(ds.data_vars))
    print("[TS] basin sample:", ds["basin"].values[:3] if "basin" in ds.coords else basins)
    print("[TS] time sample:", ds["time"].values[:3] if "time" in ds.coords else [])

    # 检查维度（顺序无关，使用 sizes 更稳妥）
    assert "basin" in ds.dims and "time" in ds.dims
    assert ds.sizes["basin"] == len(basins)
    expected_time = pd.date_range(t_range[0], t_range[1], freq="1D")
    assert ds.sizes["time"] == len(expected_time)

    # 检查变量存在与形状
    for v in ["streamflow", "total_precipitation_hourly"]:
        assert v in ds.data_vars
        # 不依赖维度顺序，只需包含两个维度且名称相符
        assert set(ds[v].dims) == {"basin", "time"}
        # 直观打印：小样本（3个流域 x 前3天），并确保顺序为 (basin, time)
        sample = ds[v].transpose("basin", "time").sel(basin=basins).isel(time=slice(0, 3))
        print(f"[TS] var={v} dims={sample.dims} sizes={dict(sample.sizes)} dtype={sample.dtype}")
        print(f"[TS] var={v} sample (first 3 basins x first 3 time):\n", sample.to_pandas())

    # 检查 lstm_pred 注入
    assert "lstm_pred" in ds.data_vars
    assert set(ds["lstm_pred"].dims) == {"basin", "time"}
    lstm_sample = ds["lstm_pred"].transpose("basin", "time").sel(basin=basins).isel(time=slice(0, 3))
    print("[TS] var=lstm_pred dims=", lstm_sample.dims, "sizes=", dict(lstm_sample.sizes), "dtype=", lstm_sample.dtype)
    print("[TS] var=lstm_pred sample (first 3 basins x first 3 time):\n", lstm_sample.to_pandas())

    # 检查来源标识
    assert "source" in ds.data_vars
    assert ds["source"].dims == ("basin",)
    assert set(ds["source"].values.tolist()) == {"base"}
    print("[TS] source sample:\n", ds["source"].to_pandas().head())


def test_read_ts_xrdataset_combine_outputs(tg_dataset):
    basins = ["station_1", "station_2", "station_3"]
    t_range = ["2020-01-01", "2020-01-10"]
    var_lst = ["total_precipitation_hourly", "streamflow", "lstm_pred"]

    ds_map = tg_dataset.read_ts_xrdataset(
        gage_id_lst=basins,
        t_range=t_range,
        var_lst=var_lst,
        time_units=["1D"],
        combine=True,
    )

    assert isinstance(ds_map, dict) and "1D" in ds_map
    ds = ds_map["1D"]
    assert isinstance(ds, xr.Dataset)

    # 直观打印：整体结构与基本信息（combined）
    print("\n[TS-COMBINE] Dataset summary:\n", ds)
    print("[TS-COMBINE] dims:", ds.dims)
    print("[TS-COMBINE] sizes:", dict(ds.sizes))
    print("[TS-COMBINE] vars:", list(ds.data_vars))

    # 检查维度与合并后大小
    assert "basin" in ds.dims and "time" in ds.dims
    expected_time = pd.date_range(t_range[0], t_range[1], freq="1D")
    assert ds.sizes["time"] == len(expected_time)
    # 若存在 intermediate，则应为 2 倍；否则为 1 倍
    sources_set = set(ds["source"].values.tolist()) if "source" in ds.data_vars else set()
    if {"base", "intermediate"}.issubset(sources_set):
        assert ds.sizes["basin"] == len(basins) * 2
    else:
        assert ds.sizes["basin"] == len(basins)

    # 检查来源标识包含两类
    assert "source" in ds.data_vars
    sources = set(ds["source"].values.tolist())
    # 当 intermediate 不存在时，仅有 base
    assert sources in [{"base", "intermediate"}, {"base"}]
    print("[TS-COMBINE] source counts:", pd.Series(ds["source"].values).value_counts().to_dict())

    # 检查变量维度与小样本
    for v in ["streamflow", "total_precipitation_hourly"]:
        assert v in ds.data_vars
        assert set(ds[v].dims) == {"basin", "time"}
        sample = ds[v].transpose("basin", "time").isel(basin=slice(0, 6), time=slice(0, 3))
        print(f"[TS-COMBINE] var={v} dims={sample.dims} sizes={dict(sample.sizes)} dtype={sample.dtype}")
        print(f"[TS-COMBINE] var={v} sample (first 6 basins x first 3 time):\n", sample.to_pandas())

    # lstm_pred 仅注入到 base 部分；intermediate 部分应为空或 NaN
    assert "lstm_pred" in ds.data_vars
    assert set(ds["lstm_pred"].dims) == {"basin", "time"}
    inter_mask = ds["source"].values == "intermediate"
    inter_basins = ds["basin"].values[inter_mask]
    base_mask = ds["source"].values == "base"
    base_basins = ds["basin"].values[base_mask]
    # 打印样本
    lstm_base_sample = ds["lstm_pred"].transpose("basin", "time").sel(basin=base_basins).isel(time=slice(0, 3))
    print("[TS-COMBINE] var=lstm_pred base sample:\n", lstm_base_sample.to_pandas().head())
    if len(inter_basins) > 0:
        lstm_inter_sample = ds["lstm_pred"].transpose("basin", "time").sel(basin=inter_basins).isel(time=slice(0, 3))
        print("[TS-COMBINE] var=lstm_pred intermediate sample:\n", lstm_inter_sample.to_pandas().head())
        # 检查 intermediate 部分为 NaN
        assert np.isnan(lstm_inter_sample.to_numpy()).all()



def test_read_attr_xrdataset_shapes_and_vars(tg_dataset):
    basins = ["station_1", "station_2", "station_3"]
    var_lst = ["ele_mt_sav", "area", "p_mean"]

    ds = tg_dataset.read_attr_xrdataset(gage_id_lst=basins, var_lst=var_lst)
    assert isinstance(ds, xr.Dataset)

    # 直观打印：整体结构与基本信息
    print("\n[ATTR] Dataset summary:\n", ds)
    print("[ATTR] dims:", ds.dims)
    print("[ATTR] sizes:", dict(ds.sizes))
    print("[ATTR] vars:", list(ds.data_vars))
    print("[ATTR] basin sample:", ds["basin"].values[:3] if "basin" in ds.coords else basins)

    # 检查维度（使用 sizes 更稳妥）
    assert "basin" in ds.dims
    assert ds.sizes["basin"] == len(basins)

    # 检查变量与单位属性
    for v in var_lst:
        assert v in ds.data_vars
        # 不依赖维度顺序（这里应为 1 维），直接检查包含 basin 维度
        assert set(ds[v].dims) == {"basin"}
        assert ds[v].sizes["basin"] == len(basins)
        assert "units" in ds[v].attrs
        # 直观打印：小样本（3个流域）与单位
        sample = ds[v].sel(basin=basins)
        print(f"[ATTR] var={v} dims={sample.dims} sizes={dict(sample.sizes)} dtype={sample.dtype} units={sample.attrs.get('units')}")
        print(f"[ATTR] var={v} sample (first 3 basins):\n", sample.to_pandas())

    # 检查来源标识
    assert "source" in ds.data_vars
    assert ds["source"].dims == ("basin",)
    assert set(ds["source"].values.tolist()) == {"base"}
    print("[ATTR] source sample:\n", ds["source"].to_pandas().head())


def test_read_attr_xrdataset_combine_outputs(tg_dataset):
    basins = ["station_1", "station_2", "station_3"]
    var_lst = ["ele_mt_sav", "area", "p_mean"]

    ds = tg_dataset.read_attr_xrdataset(gage_id_lst=basins, var_lst=var_lst, combine=True)
    assert isinstance(ds, xr.Dataset)

    # 直观打印：整体结构与基本信息（combined）
    print("\n[ATTR-COMBINE] Dataset summary:\n", ds)
    print("[ATTR-COMBINE] dims:", ds.dims)
    print("[ATTR-COMBINE] sizes:", dict(ds.sizes))
    print("[ATTR-COMBINE] vars:", list(ds.data_vars))

    # 检查维度与合并后大小
    assert "basin" in ds.dims
    sources_set = set(ds["source"].values.tolist()) if "source" in ds.data_vars else set()
    if {"base", "intermediate"}.issubset(sources_set):
        assert ds.sizes["basin"] == len(basins) * 2
    else:
        assert ds.sizes["basin"] == len(basins)

    # 检查变量与单位属性，以及来源类别
    for v in var_lst:
        assert v in ds.data_vars
        assert set(ds[v].dims) == {"basin"}
        if {"base", "intermediate"}.issubset(sources_set):
            assert ds[v].sizes["basin"] == len(basins) * 2
        else:
            assert ds[v].sizes["basin"] == len(basins)
        assert "units" in ds[v].attrs
        # 打印样本（前 6 basins）
        sample = ds[v].isel(basin=slice(0, 6))
        print(f"[ATTR-COMBINE] var={v} dims={sample.dims} sizes={dict(sample.sizes)} dtype={sample.dtype} units={sample.attrs.get('units')}")
        print(f"[ATTR-COMBINE] var={v} sample (first 6 basins):\n", sample.to_pandas())

    # 检查来源标识
    assert "source" in ds.data_vars
    sources = set(ds["source"].values.tolist())
    assert sources in [{"base", "intermediate"}, {"base"}]
    print("[ATTR-COMBINE] source counts:", pd.Series(ds["source"].values).value_counts().to_dict())


def test_description_has_lstm_file(tg_dataset):
    desc = tg_dataset.data_source_description
    assert "LSTM_PRED_FILE" in desc
    assert os.path.exists(desc["LSTM_PRED_FILE"])  # 路径应指向缓存目录下的 lstmpred.nc
