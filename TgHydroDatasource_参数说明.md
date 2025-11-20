# TgHydroDatasource 参数说明文档

## 构造函数参数

### TgHydroDatasource 类的入参

```python
def __init__(self, data_path, download=False, time_unit=None, dataset_name=None, **kwargs):
```

### 参数详细说明

#### 1. 必需参数

| 参数名 | 类型 | 说明 | 示例 |
|--------|------|------|------|
| `data_path` | str | 数据根目录路径，可以是本地路径或S3路径 | `/path/to/data` 或 `s3://bucket/data` |

#### 2. 可选参数

| 参数名 | 类型 | 默认值 | 说明 | 示例 |
|--------|------|--------|------|------|
| `download` | bool | False | 是否下载数据（如果数据源支持） | `True` 或 `False` |
| `time_unit` | list | ["1D"] | 时间单位列表，支持的单位有：["1h", "3h", "1D", "8D"] | `["1D"]` 或 `["1h", "3h"]` |
| `dataset_name` | str | None | 数据集名称，用于缓存文件命名等 | `"my_tg_dataset"` |
| `**kwargs` | dict | {} | 其他关键字参数，会传递给父类 | `{"version": "v1.0"}` |

#### 3. kwargs 中支持的参数

| 参数名 | 类型 | 默认值 | 说明 | 示例 |
|--------|------|--------|------|------|
| `version` | str | None | 数据集版本号 | `"v2.0"` |

## 参数传递机制

### 1. 继承关系
```
HydroData (基类)
    ↓
SelfMadeHydroDataset (父类)
    ↓
TgHydroDatasource (子类)
```

### 2. 参数传递流程

1. **TgHydroDatasource** 接收所有参数
2. 通过 `super().__init__(data_path, download, time_unit, dataset_name, **kwargs)` 调用父类
3. **SelfMadeHydroDataset** 处理自己的参数，再通过 `super().__init__(data_path)` 调用基类
4. **HydroData** 基类只接收 `data_path` 参数

### 3. 参数处理详情

#### 在 SelfMadeHydroDataset 中：
```python
def __init__(self, data_path, download=False, time_unit=None, dataset_name=None, **kwargs):
    # 参数验证和默认值设置
    if time_unit is None:
        time_unit = ["1D"]
    if any(unit not in ["1h", "3h", "1D", "8D"] for unit in time_unit):
        raise ValueError("time_unit must be one of ['1h', '3h', '1D', '8D']")
    
    # 设置存储类型（本地或minio）
    self.head = "minio" if "s3://" in data_path else "local"
    
    # 调用基类
    super().__init__(data_path)
    
    # 设置实例属性
    self.data_source_description = self.set_data_source_describe()
    self.time_unit = time_unit
    self.dataset_name = dataset_name
    self.version = kwargs.get("version", None)  # 从kwargs中提取version
    
    # 其他初始化
    if download:
        self.download_data_source()
    self.camels_sites = self.read_site_info()
```

#### 在 TgHydroDatasource 中：
```python
def __init__(self, data_path, download=False, time_unit=None, dataset_name=None, **kwargs):
    # 调用父类初始化（所有参数都传递给父类）
    super().__init__(data_path, download, time_unit, dataset_name, **kwargs)
    
    # TgHydroDatasource 特有的初始化
    self.graph_dict = self._load_graph_dict()
```

## 特殊参数处理

### 1. TgHydroDatasource 没有独有参数
- TgHydroDatasource 的构造函数参数与父类 SelfMadeHydroDataset 完全相同
- 所有参数都通过 `super().__init__()` 传递给父类处理
- 子类只是在父类初始化完成后，添加了图网络结构的加载

### 2. 如果子类有独有参数的处理方式

假设 TgHydroDatasource 有独有参数，处理方式如下：

```python
def __init__(self, data_path, download=False, time_unit=None, dataset_name=None, 
             graph_config=None, **kwargs):  # graph_config 是子类独有参数
    """
    Parameters
    ----------
    graph_config : dict, optional
        图网络配置参数（子类独有）
    """
    # 处理子类独有参数
    self.graph_config = graph_config or {}
    
    # 调用父类初始化（不传递子类独有参数）
    super().__init__(data_path, download, time_unit, dataset_name, **kwargs)
    
    # 子类特有的初始化
    self.graph_dict = self._load_graph_dict()
```

## 数据目录结构要求

TgHydroDatasource 期望的数据目录结构：

```
data_path/
├── attributes/           # 流域属性数据（父类要求）
│   └── attributes.csv
├── timeseries/          # 时间序列数据（父类要求）
│   ├── 1D/             # 日数据
│   ├── 3h/             # 3小时数据
│   └── *_units_info.json
├── shapes/              # 流域形状文件（父类要求）
│   └── *.shp
└── tggnn/               # TG-GNN专用数据（子类要求）
    ├── lstmpred.nc      # LSTM预测数据
    └── graph_dict.json  # 图网络结构
```

## 使用示例

### 1. 基本使用
```python
from hydrodatasource.reader.data_source import TgHydroDatasource

# 最简单的使用方式
tg_datasource = TgHydroDatasource(data_path="/path/to/data")
```

### 2. 完整参数使用
```python
tg_datasource = TgHydroDatasource(
    data_path="/path/to/data",
    download=False,
    time_unit=["1D", "3h"],
    dataset_name="my_tg_dataset",
    version="v1.0"
)
```

### 3. S3路径使用
```python
tg_datasource = TgHydroDatasource(
    data_path="s3://my-bucket/tg-data",
    time_unit=["1D"],
    dataset_name="s3_tg_dataset"
)
```

## 注意事项

1. **time_unit 限制**：只支持 ["1h", "3h", "1D", "8D"] 四种时间单位
2. **数据路径**：支持本地路径和S3路径，S3路径需要以 "s3://" 开头
3. **图网络文件**：必须存在 `tggnn/graph_dict.json` 文件，否则初始化会失败
4. **继承关系**：所有父类的方法和属性都可以使用
5. **参数传递**：kwargs 中的参数会传递给父类，可以用于扩展功能