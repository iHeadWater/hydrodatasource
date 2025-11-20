# read_lstm_predictions 方法参数传递说明

## 方法签名

```python
def read_lstm_predictions(self, object_ids=None, t_range_list=None, **kwargs):
```

## 参数传递方式

### 1. 直接调用方式

用户可以直接调用`read_lstm_predictions`方法：

```python
from hydrodatasource.reader.data_source import TgHydroDatasource

# 初始化数据源
tg_datasource = TgHydroDatasource(data_path="/path/to/data")

# 直接调用read_lstm_predictions
lstm_data = tg_datasource.read_lstm_predictions(
    object_ids=['basin_1', 'basin_2', 'basin_3'],
    t_range_list=['2020-01-01', '2020-12-31']
)
```

### 2. 通过read_timeseries_with_lstm间接调用

更常见的使用方式是通过`read_timeseries_with_lstm`方法间接调用：

```python
# 通过read_timeseries_with_lstm调用
timeseries_data, lstm_data = tg_datasource.read_timeseries_with_lstm(
    object_ids=['basin_1', 'basin_2', 'basin_3'],  # 这个参数会传递给read_lstm_predictions
    t_range=['2020-01-01', '2020-12-31'],          # 这个参数会传递给read_lstm_predictions的t_range_list
    var_lst=['streamflow', 'precipitation']
)
```

## 参数详细说明

### object_ids 参数

| 属性 | 说明 |
|------|------|
| **类型** | `list` 或 `None` |
| **默认值** | `None` |
| **作用** | 指定要读取的流域ID列表 |
| **传递方式** | 直接传递或通过`read_timeseries_with_lstm`的`object_ids`参数传递 |

**使用示例：**
```python
# 方式1：直接传递
lstm_data = tg_datasource.read_lstm_predictions(
    object_ids=['basin_001', 'basin_002', 'basin_003']
)

# 方式2：通过read_timeseries_with_lstm传递
timeseries_data, lstm_data = tg_datasource.read_timeseries_with_lstm(
    object_ids=['basin_001', 'basin_002', 'basin_003']  # 会传递给read_lstm_predictions
)
```

**内部处理逻辑：**
```python
# 在read_lstm_predictions方法内部
if object_ids is not None:
    lstm_data = lstm_data.sel(basin=object_ids)  # 使用xarray的sel方法筛选流域
```

### t_range_list 参数

| 属性 | 说明 |
|------|------|
| **类型** | `list` 或 `None` |
| **默认值** | `None` |
| **格式** | `[start_time, end_time]` |
| **作用** | 指定要读取的时间范围 |
| **传递方式** | 直接传递或通过`read_timeseries_with_lstm`的`t_range`参数传递 |

**使用示例：**
```python
# 方式1：直接传递
lstm_data = tg_datasource.read_lstm_predictions(
    t_range_list=['2020-01-01', '2020-12-31']
)

# 方式2：通过read_timeseries_with_lstm传递
timeseries_data, lstm_data = tg_datasource.read_timeseries_with_lstm(
    t_range=['2020-01-01', '2020-12-31']  # 会传递给read_lstm_predictions的t_range_list
)
```

**内部处理逻辑：**
```python
# 在read_lstm_predictions方法内部
if t_range_list is not None:
    start_time = pd.to_datetime(t_range_list[0])
    end_time = pd.to_datetime(t_range_list[1])
    lstm_data = lstm_data.sel(time=slice(start_time, end_time))  # 使用xarray的时间切片
```

## 参数传递流程图

```
用户调用
    ↓
┌─────────────────────────────────────────────────────────────┐
│ 方式1: 直接调用                                              │
│ tg_datasource.read_lstm_predictions(                       │
│     object_ids=['basin_1', 'basin_2'],                     │
│     t_range_list=['2020-01-01', '2020-12-31']              │
│ )                                                           │
└─────────────────────────────────────────────────────────────┘
    ↓
read_lstm_predictions方法接收参数
    ↓
处理和筛选数据

┌─────────────────────────────────────────────────────────────┐
│ 方式2: 通过read_timeseries_with_lstm间接调用                 │
│ tg_datasource.read_timeseries_with_lstm(                   │
│     object_ids=['basin_1', 'basin_2'],                     │
│     t_range=['2020-01-01', '2020-12-31'],                  │
│     var_lst=['streamflow']                                  │
│ )                                                           │
└─────────────────────────────────────────────────────────────┘
    ↓
read_timeseries_with_lstm方法内部调用
    ↓
self.read_lstm_predictions(
    object_ids=object_ids,      # 直接传递
    t_range_list=t_range        # 参数名映射：t_range → t_range_list
)
    ↓
read_lstm_predictions方法接收参数
    ↓
处理和筛选数据
```

## 参数验证和错误处理

### object_ids 验证
- 如果传入的`object_ids`在数据集中不存在，xarray会抛出`KeyError`
- 建议先使用`tg_datasource.read_object_ids()`获取可用的流域ID

### t_range_list 验证
- 时间格式会通过`pd.to_datetime()`自动解析
- 支持多种时间格式：'2020-01-01', '2020/01/01', '20200101'等
- 如果时间范围超出数据集范围，会返回空的时间切片

## 完整使用示例

```python
from hydrodatasource.reader.data_source import TgHydroDatasource

# 初始化
tg_datasource = TgHydroDatasource(data_path="/path/to/data")

# 获取可用的流域ID
available_basins = tg_datasource.read_object_ids()
print(f"可用流域: {available_basins[:5]}")

# 选择要处理的流域和时间范围
selected_basins = available_basins[:3]
time_range = ['2020-01-01', '2020-12-31']

# 方式1：直接调用read_lstm_predictions
lstm_data_direct = tg_datasource.read_lstm_predictions(
    object_ids=selected_basins,
    t_range_list=time_range
)

# 方式2：通过read_timeseries_with_lstm调用
timeseries_data, lstm_data_indirect = tg_datasource.read_timeseries_with_lstm(
    object_ids=selected_basins,
    t_range=time_range,
    var_lst=['streamflow', 'precipitation']
)

# 两种方式得到的lstm_data应该是相同的
print(f"直接调用结果: {lstm_data_direct.dims}")
print(f"间接调用结果: {lstm_data_indirect.dims}")
```

## 注意事项

1. **参数名映射**：`read_timeseries_with_lstm`中的`t_range`参数会映射为`read_lstm_predictions`中的`t_range_list`参数

2. **数据筛选顺序**：先按流域筛选，再按时间筛选

3. **性能考虑**：如果数据集很大，建议先筛选时间范围再筛选流域，以提高性能

4. **错误处理**：建议在调用前验证参数的有效性，避免运行时错误