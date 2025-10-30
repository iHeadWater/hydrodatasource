# StationHydroDataset 使用指南

## 概述

`StationHydroDataset` 是 `SelfMadeHydroDataset` 的扩展类，专门用于处理包含站点数据的水文数据集。它支持读取、缓存和分析站点时序数据，以及流域-站点关系信息。

## 数据结构

期望的数据目录结构如下：

```
your_dataset/
├── attributes/                      # 流域属性数据
│   └── attributes.csv
├── shapes/                          # 空间数据
│   ├── basins.shp
│   └── basinoutlets.shp
├── timeseries/                      # 流域时序数据
│   ├── 1D/
│   └── 3h/
└── stations/                        # 站点数据（新增）
    ├── 1D/                          # 日数据
    │   ├── chn_dllg_20810200.csv
    │   ├── chn_dllg_20810400.csv
    │   └── ...
    ├── 3h/                          # 3小时数据
    │   ├── chn_dllg_20810200.csv
    │   ├── chn_dllg_20810400.csv
    │   └── ...
    └── basin_station_info/          # 流域-站点关系信息
        ├── all_basin_station_mapping.csv
        ├── basin_summary.csv
        ├── basin_20810200_stations.csv
        ├── basin_21401550_stations.csv
        ├── adjacency_20810200_True.csv
        ├── adjacency_21401550_True.csv
        └── ...
```

## 基本使用

### 1. 初始化数据集

```python
from hydrodatasource.reader.data_source import StationHydroDataset

# 创建数据集实例
dataset = StationHydroDataset(
    data_path="/path/to/your/dataset",
    time_unit=["1D", "3h"],           # 支持的时间单位
    dataset_name="my_station_dataset",  # 数据集名称
    version="v1.0",                    # 版本号
    offset_to_utc=True                 # 是否转换为UTC时间
)
```

### 2. 读取站点信息

```python
# 读取基本站点信息和流域-站点映射
mapping_data, summary_data = dataset.read_station_info()

# 获取所有站点ID
all_station_ids = dataset.read_station_object_ids()

# 获取特定流域的站点
basin_stations = dataset.get_stations_by_basin("20810200")
```

### 3. 读取详细站点信息

```python
# 读取特定流域的详细站点信息
station_details = dataset.read_basin_stations("20810200")

# 读取流域内站点的邻接矩阵
adjacency_matrix = dataset.read_basin_adjacency("20810200")
```

### 4. 读取站点时序数据

```python
# 读取站点时序数据
station_data = dataset.read_station_timeseries(
    station_ids=["chn_dllg_20810200", "chn_dllg_20810400"],
    t_range_list=["2020-01-01", "2020-12-31"],
    relevant_cols=["streamflow", "water_level"],
    time_units=["1D"]
)

# 返回格式：{"1D": numpy.ndarray}
# 数组形状：(stations, time_points, variables)
```

## 数据缓存功能

### 1. 缓存站点时序数据

```python
# 缓存所有站点的时序数据到NetCDF文件
dataset.cache_station_timeseries_xrdataset(
    batchsize=100,                    # 批处理大小
    time_units=["1D", "3h"],         # 要缓存的时间单位
    start0101_freq=False             # 是否使用01-01开始的频率
)
```

### 2. 缓存站点信息

```python
# 缓存站点信息和流域-站点关系
dataset.cache_station_info_xrdataset()
```

### 3. 一次性缓存所有数据

```python
# 缓存所有站点相关数据
dataset.cache_all_station_data(
    batchsize=100,
    time_units=["1D"]
)
```

## 从缓存读取数据

### 1. 读取缓存的时序数据

```python
# 从缓存的NetCDF文件读取站点时序数据
cached_data = dataset.read_station_ts_xrdataset(
    station_id_lst=["chn_dllg_20810200", "chn_dllg_20810400"],
    t_range=["2020-01-01", "2020-12-31"],
    var_lst=["streamflow", "water_level"],
    time_units=["1D"]
)

# 返回格式：{"1D": xarray.Dataset}
for time_unit, ds in cached_data.items():
    print(f"Variables: {list(ds.data_vars)}")
    print(f"Stations: {len(ds.station)}")
    print(f"Time points: {len(ds.time)}")
```

### 2. 读取缓存的站点信息

```python
# 从缓存读取站点信息
mapping_ds, summary_ds = dataset.read_station_info_xrdataset()

# mapping_ds: 包含流域-站点映射的xarray.Dataset
# summary_ds: 包含流域汇总信息的xarray.Dataset
```

## 文件格式要求

### 1. 站点时序数据文件格式

站点CSV文件应包含以下列：

```csv
time,streamflow,water_level,temperature
2020-01-01,10.5,2.3,5.2
2020-01-02,12.1,2.4,5.8
...
```

### 2. 流域-站点映射文件格式

`all_basin_station_mapping.csv`:

```csv
basin_id,station_id
20810200,chn_dllg_20810200
20810200,chn_dllg_20810201
21401550,chn_dllg_21401550
...
```

### 3. 流域汇总信息文件格式

`basin_summary.csv`:

```csv
basin_id,basin_name,station_count,area_km2
20810200,Basin A,15,1250.5
21401550,Basin B,12,980.2
...
```

### 4. 流域站点详情文件格式

`basin_20810200_stations.csv`:

```csv
station_id,station_name,station_type,latitude,longitude,elevation
chn_dllg_20810200,Station A,streamflow,40.123,116.456,125.5
chn_dllg_20810201,Station B,water_level,40.234,116.567,130.2
...
```

### 5. 邻接矩阵文件格式

`adjacency_20810200_True.csv`:

```csv
,chn_dllg_20810200,chn_dllg_20810201,chn_dllg_20810202
chn_dllg_20810200,0,1,0
chn_dllg_20810201,1,0,1
chn_dllg_20810202,0,1,0
```

## 完整示例

```python
from hydrodatasource.reader.data_source import StationHydroDataset

# 1. 初始化数据集
dataset = StationHydroDataset(
    data_path="/path/to/your/dataset",
    time_unit=["1D", "3h"],
    dataset_name="my_stations",
    version="v1.0",
    offset_to_utc=True
)

# 2. 读取站点信息
mapping_data, summary_data = dataset.read_station_info()
print(f"Total basins: {len(summary_data)}")

# 3. 分析特定流域
basin_id = "20810200"
stations = dataset.get_stations_by_basin(basin_id)
print(f"Stations in {basin_id}: {len(stations)}")

# 4. 读取站点详情
station_details = dataset.read_basin_stations(basin_id)
adjacency = dataset.read_basin_adjacency(basin_id)

# 5. 读取时序数据
station_data = dataset.read_station_timeseries(
    station_ids=stations[:5],  # 前5个站点
    t_range_list=["2020-01-01", "2020-12-31"],
    relevant_cols=["streamflow", "water_level"],
    time_units=["1D"]
)

# 6. 缓存数据
dataset.cache_all_station_data(batchsize=50, time_units=["1D"])

# 7. 从缓存读取数据
cached_data = dataset.read_station_ts_xrdataset(
    station_id_lst=stations[:5],
    t_range=["2020-01-01", "2020-12-31"],
    var_lst=["streamflow", "water_level"],
    time_units=["1D"]
)

# 8. 分析数据
for time_unit, ds in cached_data.items():
    print(f"Dataset shape: {ds.dims}")
    print(f"Variables: {list(ds.data_vars)}")
    
    # 计算平均值
    mean_streamflow = ds["streamflow"].mean(dim="time")
    print(f"Mean streamflow by station: {mean_streamflow.values}")
```

## 注意事项

1. **文件路径**: 支持本地路径和S3路径（以"s3://"开头）
2. **时间处理**: 支持UTC转换，特别是对于中国时区数据
3. **批处理**: 大数据集使用批处理以避免内存问题
4. **缓存管理**: 缓存文件存储在`CACHE_DIR`目录中
5. **错误处理**: 文件不存在时会抛出相应异常

## 性能优化建议

1. **批处理大小**: 根据内存大小调整`batchsize`参数
2. **时间范围**: 只缓存需要的时间范围数据
3. **变量选择**: 只读取需要的变量以减少内存使用
4. **并行处理**: 可以针对不同流域并行处理数据

## 扩展功能

该类可以进一步扩展以支持：

- 站点数据质量检查
- 站点数据插值和填充
- 站点网络分析
- 时序数据统计分析
- 多源数据融合

通过继承`StationHydroDataset`类，您可以添加特定于您的数据集的功能和分析方法。 