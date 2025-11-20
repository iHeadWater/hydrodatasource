# hydrodatasource

[![image](https://img.shields.io/pypi/v/hydrodatasource.svg)](https://pypi.python.org/pypi/hydrodatasource) [![image](https://img.shields.io/conda/vn/conda-forge/hydrodatasource.svg)](https://anaconda.org/conda-forge/hydrodatasource)

-   免费软件：BSD 许可证
-   文档：https://iHeadWater.github.io/hydrodatasource

## 概述

尽管像 [hydrodataset](https://github.com/OuyangWenyu/hydrodataset) 这样的库可以用来访问标准化的公共水文数据集（例如 CAMELS），但在实际工作中，我们经常需要处理那些非即用格式的数据。这包括非公开的行业数据、地方当局提供的数据，或为特定研究项目编制的自定义数据集。

**`hydrodatasource`** 正是为了解决这一问题而设计的。它提供了一个灵活的框架，用于读取、处理和清洗这些自定义数据集，为水文建模和分析做准备。

该框架的核心是 `SelfMadeHydroDataset` 类，它允许您通过将数据组织成一个简单的、预定义的目录结构来轻松访问您自己的数据。

## 使用 `SelfMadeHydroDataset` 读取自定义数据集

这是 `hydrodatasource` 的主要用例。如果您有自己的流域级别时间序列和属性数据，您可以使用此类无缝加载它。

### 1. 准备您的数据目录

首先，将您的数据组织成以下文件夹结构：

```
/path/to/your_data_root/
    └── my_custom_dataset/              # 您的数据集名称
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
-   **`timeseries/1D/`**: 每个时间分辨率一个文件夹（例如，`1D` 表示日尺度，`3h` 表示 3 小时尺度）。在内部，每个 CSV 文件应包含单个流域的时间序列数据，并以其 `basin_id` 命名。
-   **`timeseries/1D_units_info.json`**: 定义时间序列 CSV 中每个变量单位的 JSON 文件（例如，`{"precipitation": "mm/d", "streamflow": "m3/s"}`）。

### 2. 在 Python 中读取数据

一旦您的数据被组织好，您就可以使用 `SelfMadeHydroDataset` 通过几行代码来读取它。

```python
from hydrodatasource.reader.data_source import SelfMadeHydroDataset

# 1. 定义数据父目录的路径和数据集名称
data_path = "/path/to/your_data_root/"
dataset_name = "my_custom_dataset"

# 2. 初始化读取器
# 指定您要使用的时间单位
reader = SelfMadeHydroDataset(data_path=data_path, dataset_name=dataset_name, time_unit=["1D"])

# 3. 获取所有可用流域 ID 的列表
basin_ids = reader.read_object_ids()

# 4. 定义您要加载的时间范围和变量
t_range = ["2000-01-01", "2010-12-31"]
variables_to_read = ["precipitation", "streamflow", "temperature"]

# 5. 读取时间序列数据
# 结果是一个 xarray.Dataset 的字典，以时间单位为键
timeseries_data = reader.read_ts_xrdataset(
    gage_id_lst=basin_ids,
    t_range=t_range,
    var_lst=variables_to_read,
    time_units=["1D"])

daily_data = timeseries_data["1D"]

print("数据加载成功：")
print(daily_data)

# 您也可以读取静态属性
attributes_data = reader.read_attr_xrdataset(gage_id_lst=basin_ids, var_lst=["area", "mean_elevation"])
print("\n属性：")
print(attributes_data)
```

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

对于开发人员，建议使用 `uv` 来管理环境，因为该项目具有本地依赖项（例如 `hydroutils`、`hydrodataset`）。

1.  **克隆存储库：**
    ```bash
    git clone https://github.com/iHeadWater/hydrodatasource.git
    cd hydrodatasource
    ```

2.  **使用 `uv` 同步环境：**
    此命令将安装所有依赖项，包括本地可编辑的软件包。
    ```bash
    uv sync --all-extras
    ```