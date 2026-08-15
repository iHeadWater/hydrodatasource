# hydrodatasource

[![image](https://img.shields.io/pypi/v/hydrodatasource.svg)](https://pypi.python.org/pypi/hydrodatasource) [![image](https://img.shields.io/conda/vn/conda-forge/hydrodatasource.svg)](https://anaconda.org/conda-forge/hydrodatasource)

-   免费软件：BSD 许可证
-   文档：https://iHeadWater.github.io/hydrodatasource

## 概述

尽管像 [hydrodataset](https://github.com/OuyangWenyu/hydrodataset) 这样的库可以用来访问标准化的公共水文数据集（例如 CAMELS），但在实际工作中，我们经常需要处理那些非即用格式的数据。这包括非公开的行业数据、地方当局提供的数据，或为特定研究项目编制的自定义数据集。

**`hydrodatasource`** 正是为了解决这一问题而设计的。它提供了一个灵活的框架，用于读取、处理和清洗这些自定义数据集，为水文建模和分析做准备。

`hydrodatasource` 采用统一的 **URI-only** 设计：所有读取器都通过指向一个目录或 `s3://` URI 来构造，一行 `open_dataset()` 工厂即可将数据集 id 解析为可用的读取器。该设计与 [hydrodataset](https://github.com/OuyangWenyu/hydrodataset) 兼容。

## 快速上手

如果您有**已注册的数据集**和 `~/hydro_setting.yml`（见[配置](#配置)），可一行解析并打开：

```python
from hydrodatasource.configs.data_resolver import open_dataset

# "songliao_event" 是 hydrodatasource 注册的数据集（洪水事件读取器）
ds = open_dataset("songliao_event")
```

要**按路径打开自定义数据集**，请直接构造读取器：

```python
from hydrodatasource.reader.data_source import SelfMadeHydroDataset

reader = SelfMadeHydroDataset(uri="/path/to/my_dataset", time_unit=["1D"])
```

## 读取自定义数据集

这是 `hydrodatasource` 的主要用例。如果您有自己的流域级别时间序列和属性数据，您可以使用 `SelfMadeHydroDataset` 无缝加载它。

### 1. 准备您的数据目录

首先，将您的数据组织成以下文件夹结构：

```
/path/to/my_dataset/
├── attributes/
│   └── attributes.csv
├── shapes/
│   └── basins.shp
└── timeseries/
    ├── 1D/                     # 每个时间分辨率一个子文件夹（例如，日尺度）
    │   ├── basin_01.csv
    │   ├── basin_02.csv
    │   └── ...
    └── 1D_units_info.json      # 包含单位信息的 JSON 文件
```

-   **`attributes/attributes.csv`**: 包含静态流域属性（例如，面积、平均高程）的 CSV 文件。必须包含一个 `basin_id` 列，该列与 `timeseries` 文件夹中的文件名匹配。
-   **`shapes/basins.shp`**: 包含每个流域多边形几何的 shapefile 文件。
-   **`timeseries/1D/`**: 每个时间分辨率一个文件夹（`1h`、`3h`、`1D`、`8D`、`1M`）。在内部，每个 CSV 文件应包含单个流域的时间序列数据，并以其 `basin_id` 命名。
-   **`timeseries/1D_units_info.json`**: 定义时间序列 CSV 中每个变量单位的 JSON 文件（例如，`{"precipitation": "mm/d", "streamflow": "m3/s", "temperature": "degC"}`）。您读取的每个变量都必须在此列出。

扩展数据集可选用以下可选目录：

-   **`intermediate/`** — 区间流域数据（含拓扑关系），供 TG 流域读取器使用。
-   **`stations/`** — 测站数据与邻接矩阵，供站点读取器使用。
-   **`forecasts/`** — 预报时间序列，供预报读取器使用。

### 2. 在 Python 中读取数据

一旦您的数据被组织好，将 URI-only 读取器指向它：

```python
from hydrodatasource.reader.data_source import SelfMadeHydroDataset

# 将数据集目录的绝对路径（或 s3:// URI）作为 uri 传入
reader = SelfMadeHydroDataset(uri="/path/to/my_dataset", time_unit=["1D"])

# 获取所有可用流域 ID 的列表
basin_ids = reader.read_object_ids()

# 定义您要加载的时间范围和变量
t_range = ["2000-01-01", "2010-12-31"]
variables_to_read = ["precipitation", "streamflow", "temperature"]

# 读取时间序列数据（以时间单位为键的 xarray.Dataset 字典）
timeseries_data = reader.read_ts_xrdataset(
    gage_id_lst=basin_ids,
    t_range=t_range,
    var_lst=variables_to_read,
    time_units=["1D"],
)

daily_data = timeseries_data["1D"]

print("数据加载成功：")
print(daily_data)

# 静态属性同样易于读取
attributes_data = reader.read_attr_xrdataset(gage_id_lst=basin_ids, var_lst=["area", "mean_elevation"])
print("\n属性：")
print(attributes_data)
```

> **关于旧 API 的说明。** 早期版本接受 `data_path=` / `dataset_name=` 构造参数。
> 它们已被统一的 `uri=` 接口取代，现在会抛出 `ValueError`。

## 读取器别名

所有 `hydrodatasource` 读取器都注册在 `READER_ALIASES` 中，供下游项目（如 hydromodel）与 hydrodataset 的别名一起使用：

| 别名 | 类 | 目录约定 |
|-------|-------|-----|
| `selfmade` | `SelfMadeHydroDataset` | 标准数据集（`attributes/`、`timeseries/`、`shapes/`） |
| `longterm` | `LongTermDataset` | 支持长期序列的自制数据集 |
| `forecast` | `SelfMadeForecastDataset` | 标准 + `forecasts/` |
| `station` | `StationHydroDataset` | 标准 + `stations/` |
| `tghydro` | `TgHydroDatasource` | 标准 + `intermediate/` + LSTM 预测 |
| `floodevent` | `FloodEventDatasource` | 带逐流域事件标记的洪水事件数据 |
| `gages` | `Gages` | GAGES-II 公共数据集 |
| `grdc` | `Grdc` | GRDC 公共数据集 |
| `rainfall` | `RainfallReader` | 清洗后的站点降雨 |
| `crd` | `Crd` | 中国水库数据库 |
| `rsvrinflow` | `RsvrInflowReader` | 水库入库流量数据 |

`hydrodataset` 的公共数据集（如 `camels_us`）也可通过同一个 `open_dataset()` / `resolve_data_path()` 接口解析。

## 配置

`hydrodatasource` 读取共享的 `~/hydro_setting.yml`（与 `hydrodataset`、`hydromodel` 共用），使用统一的 `storage.*` 格式：

```yaml
storage:
  default_source: local      # 'local' 或 'cloud' — 未指定 source 时的回退值
  local:
    root: 'D:\data\hydrodatasource'   # 主数据根目录
  cache: data\cache
  s3:                        # 可选 — 云端（MinIO/S3）访问
    endpoint_url: 'http://minio:9000'
    key: 'access_key'
    secret: 'secret_key'
    bucket: hydro-data
    prefix: hydromodel
```

-   `default_source: local` 时，`resolve_data_path()` / `open_dataset()` 针对 `storage.local.root` 解析。
-   `default_source: cloud` 时，针对 `storage.s3` 解析。
-   若 `~/hydro_setting.yml` 缺失，将使用默认根目录 `~/hydrodatasource_data`（带警告）。

## 其他功能

除了读取数据，`hydrodatasource` 还包括用于以下功能的模块：

-   **`processor`**: 执行高级计算，如识别降雨径流事件（`dmca_esr.py`）和计算站点数据的流域平均降雨量（`basin_mean_rainfall.py`）。
-   **`cleaner`**: 清洗原始时间序列数据。这包括平滑嘈杂的径流数据、校正降雨和水位记录中的异常，以及反算水库入库流量的工具。

这些模块的用法在[API 参考](https://iHeadWater.github.io/hydrodatasource/api)中有描述。我们将来会添加更多示例。

## 安装

对于标准使用，请从 PyPI 安装软件包：

```bash
pip install hydrodatasource
```

### 开发设置

对于开发人员，建议使用 `uv` 来管理环境，因为该项目具有本地依赖项（例如 `hydroutils>=0.2.0`、`hydrodataset>=0.3.0`）。

1.  **克隆存储库：**
    ```bash
    git clone https://github.com/iHeadWater/hydrodatasource.git
    cd hydrodatasource
    ```

2.  **使用 `uv` 同步环境**（安装全部 extras，含 dev 工具）：
    ```bash
    uv sync --all-extras
    ```
