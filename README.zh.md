<!--
 * @Author: Wenyu Ouyang
 * @Date: 2023-10-25 14:43:12
 * @LastEditTime: 2025-01-09 16:58:58
 * @LastEditors: Wenyu Ouyang
 * @Description: Chinese version README
 * @FilePath: \hydrodatasource\README.zh.md
 * Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
-->
# hydrodatasource

[![image](https://img.shields.io/pypi/v/hydrodatasource.svg)](https://pypi.python.org/pypi/hydrodatasource)
<!-- [![image](https://img.shields.io/conda/vn/conda-forge/hydrodatasource.svg)](https://anaconda.org/conda-forge/hydrodatasource) -->

-   Free software: MIT license
-   Documentation: <https://hydrodatasource.readthedocs.io/en/latest/>
 
## 概述

在水文领域，尽管已有大量公开的流域水文数据集，但仍存在以下问题：

- 数据集一次整理之后，后续数据往往没有及时整理进去；
- 现有数据集未覆盖的数据还不少；
- 一些非公开的数据无法直接共享。

为了解决这些问题，**hydrodatasource** 提供了一个框架，用于组织和管理这些数据集，从而在以流域为基本单元的科研和生产环境中更高效地利用数据。

该仓库与 [hydrodataset](https://github.com/OuyangWenyu/hydrodataset) 配合使用，后者专注于公开数据集，而 **hydrodatasource** 则专注于整合更广泛的数据资源，包括非公开和自定义数据源。

## 数据分类及来源

**hydrodatasource** 处理的数据主要分为三类：

### A 类数据（公开数据）

这些数据按照预定义的格式进行组织和管理，包括：

1. **GIS 数据集**：地理矢量数据，如流域边界、站点等 shapefile 文件。
2. **网格数据集**：如 ERA5Land、GPM、AIFS 等栅格数据，这些数据存储在 MinIO 数据库中。

### B 类数据（非公开或行业数据）

这些数据通常具有专有性或保密性，需借助特定工具进行格式化和整合，包括：

1. **自定义站点数据**：用户准备的站点数据，按照标准格式处理后转化为 NetCDF 格式。
2. **行业数据集**：经过整合与格式化的专业数据。

### 自建数据集

基于这两类数据，还整理出一类**自建水文数据集**，即基于约定的标准格式，为特定科研需求构建的数据集。

## 功能与特点

### 统一的数据管理

**hydrodatasource** 提供了一个标准化的方法，用于：

- 按照预定义的约定结构化数据集。
- 将多种数据源整合到统一的框架中。
- 支持水文建模的数据访问和处理。

### 兼容本地和云端资源

- **公开数据**：支持数据格式转换和本地文件操作。
- **非公开数据**：提供工具以格式化和整合用户自备数据。
- **MinIO 集成**：通过 API 支持大规模网格数据的高效管理。

### 模块化设计

仓库结构设计支持多样化的工作流程，包括：

1. **A 类 GIS 数据**：提供工具以组织和访问 GIS 数据集。
2. **A 类网格数据**：通过 MinIO 支持大规模网格数据管理。
3. **B 类数据**：提供自定义工具以清洗处理站点、水库、流域各类时序数据。
4. **自建数据集**：支持定义的标准数据集格式数据的读取。

### 其他交互关系

**hydrodatasource** 与以下组件交互：

- [**hydrodataset**](https://github.com/OuyangWenyu/hydrodataset)：为hydrodatasource提供访问公开数据集的必要支持。
- [**HydroDataCompiler**](https://github.com/iHeadWater/HydroDataCompiler)：支持非公开和自定义数据的半自动处理，暂时未公开
- [**MinIO 数据库**](http://10.48.0.86:9001/)：用于 A 类网格数据的高效存储和管理，目前仅限内部局域网访问。

## 安装

可以通过 pip 安装该包：

```bash
pip install hydrodatasource
```

注：项目仍在研发前期，推荐以开发者模式开发使用。

## 使用方法

### 数据组织

该仓库采用以下目录结构组织数据：

```
├── datasets-origin          # 公开水文数据集
├── datasets-interim         # 自建水文数据集
├── gis-origin               # 公开 GIS 数据集
├── grids-origin             # 网格数据集
├── stations-origin          # B 类站点数据（原始）
├── stations-interim         # B 类站点数据（处理后）
├── reservoirs-origin        # B 类水库数据（原始）
├── reservoirs-interim       # B 类水库数据（处理后）
├── basins-origin            # B 类流域数据（原始）
├── basins-interim           # B 类流域数据（处理后）
```

- **`origin`**：原始数据，通常来自专有来源，统一格式。
- **`interim`**：经过初步处理后可用于分析或建模的数据。

### 针对 A 类数据

1. **公开 GIS 数据**：
   - 将矢量文件存储在 `gis-origin` 文件夹中，主要包括流域边界、站点等 shapefile 数据。
   - 数据处理方法：
     ```python
     from hydrodatasource import gis
     gis.process_gis_data(input_path="gis-origin", output_path="gis-interim")
     ```

2. **网格数据集**：
   - 将原始网格数据存储在 `grids-origin`，例如 ERA5Land、GPM 等。
   - 使用 MinIO API 下载或管理存储在数据库中的网格数据：
     ```python
     from hydrodatasource import grid
     grid.download_from_minio(dataset_name="ERA5Land", save_path="grids-interim")
     ```

### 针对 B 类数据

1. **站点数据**：
   - 原始数据存储在 `stations-origin` 文件夹，处理后的数据存储在 `stations-interim`。
   - 检查站点数据的标准格式：
     ```python
     from hydrodatasource import station
     station.get_station_format()
     ```
   - 进行格式化处理：
     ```python
     station.process_station_data(input_path="stations-origin", output_path="stations-interim")
     ```

2. **水库数据**：
   - 原始水库数据存储在 `reservoirs-origin` 文件夹，清洗处理后的数据存储在 `reservoirs-interim`。
   - 针对水库数据提供特定工具函数用于整合和格式化。

3. **流域数据**：
   - 原始流域数据存储在 `basins-origin` 文件夹，处理后的数据存储在 `basins-interim`。
   - 这些数据通常包括针对流域单元的属性和空间信息，支持进一步的水文建模需求。


### **自建水文数据集**：

自建数据集保存在 `datasets-interim` 文件夹中。

按照标准格式组织，以便整合和后续模型使用。

### MinIO 数据库与数据存储

MinIO 数据库主要用于存储和管理大规模网格数据（A 类数据），例如 ERA5Land 等实时或近实时动态数据：

- 配置 MinIO 访问：
  在 `hydro_settings.yml` 文件中配置数据库相关信息。

- 上传或下载数据：
  ```python
  from hydrodatasource import minio
  minio.upload_to_minio(local_path="grids-interim/ERA5Land", dataset_name="ERA5Land")
  ```

## 结语

**hydrodatasource** 桥接了多样化的水文数据集与高级建模需求，通过提供标准化的工作流程和模块化工具，确保高效的数据管理与整合，支持科研与实际应用。
