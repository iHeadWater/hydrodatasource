# TgHydroDatasource 类输出接口说明

## 概述

`TgHydroDatasource` 类继承自 `SelfMadeHydroDataset`，提供了丰富的数据读取接口，主要用于图神经网络相关的水文建模任务。该类的输出接口可以分为以下几个类别：

## 1. 基础数据读取接口（继承自父类）

### 1.1 流域ID读取
```python
def read_object_ids(self, object_params=None) -> np.array
```
- **功能**: 读取所有可用的流域ID
- **返回**: numpy数组，包含所有流域ID
- **示例**: `basin_ids = tg_datasource.read_object_ids()`

### 1.2 时间序列数据读取
```python
def read_timeseries(self, object_ids=None, t_range_list=None, relevant_cols=None, **kwargs) -> dict
```
- **功能**: 读取原始时间序列数据
- **参数**:
  - `object_ids`: 流域ID列表
  - `t_range_list`: 时间范围 `[start_time, end_time]`
  - `relevant_cols`: 变量列表
  - `time_units`: 时间单位（如 `["1D", "3h"]`）
- **返回**: 字典，键为时间单位，值为numpy数组 `(basin_num, time_num, var_num)`

### 1.3 属性数据读取
```python
def read_attributes(self, object_ids=None, constant_cols=None, **kwargs) -> np.array
```
- **功能**: 读取流域静态属性数据
- **参数**:
  - `object_ids`: 流域ID列表
  - `constant_cols`: 属性列名列表
- **返回**: numpy数组 `(basin_num, attr_num)`

### 1.4 xarray格式数据读取
```python
def read_ts_xrdataset(self, gage_id_lst=None, t_range=None, var_lst=None, **kwargs) -> dict
```
- **功能**: 读取xarray格式的时间序列数据
- **返回**: 字典，键为时间单位，值为xarray.Dataset

```python
def read_attr_xrdataset(self, gage_id_lst=None, var_lst=None, **kwargs) -> xr.Dataset
```
- **功能**: 读取xarray格式的属性数据
- **返回**: xarray.Dataset

### 1.5 特定数据读取
```python
def read_area(self, gage_id_lst=None) -> xr.Dataset
```
- **功能**: 读取流域面积数据
- **返回**: xarray.Dataset，包含面积信息

```python
def read_mean_prcp(self, gage_id_lst=None, unit="mm/d") -> xr.Dataset
```
- **功能**: 读取平均降水数据
- **返回**: xarray.Dataset，包含降水信息

## 2. TgHydroDatasource 特有接口

### 2.1 LSTM预测数据读取
```python
def read_lstm_predictions(self, object_ids=None, t_range_list=None, **kwargs) -> xr.Dataset
```
- **功能**: 读取LSTM预测数据
- **参数**:
  - `object_ids`: 流域ID列表
  - `t_range_list`: 时间范围 `[start_time, end_time]`
- **返回**: xarray.Dataset，包含LSTM预测结果
- **数据来源**: `tggnn/lstmpred.nc` 文件

### 2.2 时间序列与LSTM数据联合读取
```python
def read_timeseries_with_lstm(self, object_ids=None, t_range=None, var_lst=None, time_units=None) -> tuple
```
- **功能**: 同时读取时间序列数据和LSTM预测数据
- **参数**:
  - `object_ids`: 流域ID列表
  - `t_range`: 时间范围 `[start_time, end_time]`
  - `var_lst`: 变量列表（默认: `['total_precipitation_hourly', 'streamflow']`）
  - `time_units`: 时间单位
- **返回**: 元组 `(timeseries_data, lstm_data)`，两个都是xarray.Dataset

### 2.3 节点属性数据读取
```python
def read_node_attributes(self, object_ids=None, selected_attrs=None) -> xr.Dataset
```
- **功能**: 读取节点静态属性数据（简化版，无PyTorch转换）
- **参数**:
  - `object_ids`: 流域ID列表
  - `selected_attrs`: 属性列表（默认: `['ele_mt_sav', 'area', 'p_mean']`）
- **返回**: xarray.Dataset，包含筛选后的属性数据

### 2.4 图网络结构生成
```python
def gen_nx_graph(self, basin_id, object_ids=None) -> tuple
```
- **功能**: 生成NetworkX图对象，确保节点索引与数据一致
- **参数**:
  - `basin_id`: 流域ID（用于查找图结构）
  - `object_ids`: 对象ID列表（可选）
- **返回**: 元组 `(NetworkX图对象, 边列表, 节点映射字典, 有效对象ID列表)`
- **数据来源**: `tggnn/graph_dict.json` 文件

### 2.5 上游节点查询
```python
def get_upstream_nodes(self, basin_id, target_node, object_ids=None) -> list
```
- **功能**: 获取指定节点的直接上游节点
- **参数**:
  - `basin_id`: 流域ID
  - `target_node`: 目标节点名称
  - `object_ids`: 所有节点名称列表（可选）
- **返回**: 直接上游节点的索引列表

## 3. 元数据和信息接口

### 3.1 数据源信息
```python
def get_name(self) -> str
```
- **功能**: 返回数据源名称
- **返回**: `"TgHydroDatasource"`

```python
def set_data_source_describe(self) -> collections.OrderedDict
```
- **功能**: 设置数据源描述，包含所有数据路径
- **返回**: 有序字典，包含数据路径信息

### 3.2 列信息获取
```python
def get_attributes_cols(self) -> np.array
```
- **功能**: 获取属性数据的列名
- **返回**: numpy数组，包含所有属性列名

```python
def get_timeseries_cols(self) -> dict
```
- **功能**: 获取时间序列数据的列名
- **返回**: 字典，键为时间单位，值为变量列表

## 4. 数据格式说明

### 4.1 返回数据类型
- **numpy.array**: 原始数据读取方法返回
- **xarray.Dataset**: 现代化数据读取方法返回，包含坐标、属性和元数据
- **dict**: 多时间单位数据返回
- **tuple**: 多种数据类型组合返回

### 4.2 坐标系统
- **basin**: 流域维度，使用流域ID作为坐标
- **time**: 时间维度，使用pandas时间索引
- **变量维度**: 根据具体数据类型而定

### 4.3 数据筛选
所有读取方法都支持：
- 按流域ID筛选（`object_ids`参数）
- 按时间范围筛选（`t_range`或`t_range_list`参数）
- 按变量筛选（`var_lst`或`selected_attrs`参数）

## 5. 使用建议

### 5.1 推荐使用的接口
- **现代化接口**: 优先使用返回xarray.Dataset的方法
- **联合读取**: 使用`read_timeseries_with_lstm`同时获取多种数据
- **图网络**: 使用`gen_nx_graph`获取完整的图结构信息

### 5.2 性能考虑
- 大数据量时使用缓存机制（`cache_xrdataset`）
- 按需读取，避免加载不必要的数据
- 合理设置时间范围和变量列表

### 5.3 错误处理
- 所有方法都包含数据验证和错误提示
- 缺失数据会给出警告信息
- 支持部分数据缺失的情况下继续处理

## 6. 示例用法

```python
from hydrodatasource.reader.data_source import TgHydroDatasource

# 初始化
tg_datasource = TgHydroDatasource(data_path="/path/to/data")

# 获取基础信息
basin_ids = tg_datasource.read_object_ids()
print(f"可用流域数量: {len(basin_ids)}")

# 读取时间序列和LSTM数据
timeseries_data, lstm_data = tg_datasource.read_timeseries_with_lstm(
    object_ids=basin_ids[:5],
    t_range=['2020-01-01', '2020-12-31'],
    var_lst=['streamflow', 'total_precipitation_hourly']
)

# 读取属性数据
attr_data = tg_datasource.read_node_attributes(
    object_ids=basin_ids[:5],
    selected_attrs=['area', 'ele_mt_sav']
)

# 生成图网络
graph, edges, node_mapping, valid_ids = tg_datasource.gen_nx_graph(
    basin_id='sanxia_60703800',
    object_ids=basin_ids[:10]
)

# 查询上游节点
upstream_nodes = tg_datasource.get_upstream_nodes(
    basin_id='sanxia_60703800',
    target_node='sanxia_60703780',
    object_ids=valid_ids
)
```

这个接口设计提供了从基础数据读取到高级图网络分析的完整功能，特别适合图神经网络相关的水文建模任务。