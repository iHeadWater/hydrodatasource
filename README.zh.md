<!--
 * @Author: Wenyu Ouyang
 * @Date: 2023-10-25 14:43:12
 * @LastEditTime: 2025-06-09 16:50:40
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

这些数据通常是论文里公开的水文领域数据集，目前包括：

- GAGES数据集
- GRDC数据集
- CRD等水库数据集

### B 类数据（非公开或行业数据）

这些数据通常具有专有性或保密性，需借助特定工具进行格式化和整合，包括：

**自定义站点数据**：用户准备的站点数据，按照标准格式处理后转化为 NetCDF 格式。

### C类自建数据集

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

### 模块化设计

仓库结构设计支持多样化的工作流程，包括：

1. **A 类数据集**：提供工具以组织和访问公开水文数据集。
2. **B 类数据**：提供自定义工具以清洗处理站点、水库、流域各类时序数据。
3. **C类自建数据集**：支持定义的标准数据集格式数据的读取。

### 其他交互关系

**hydrodatasource** 与以下组件交互：

- [**hydrodataset**](https://github.com/OuyangWenyu/hydrodataset)：为hydrodatasource提供访问公开流域水文建模数据集的必要支持。
- [**HydroDataCompiler**](https://github.com/iHeadWater/HydroDataCompiler)：支持非公开和自定义数据的半自动处理，暂时未公开

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
├── ClassA
  ├── 1st_origin
  ├── 2nd_process
├── ClassB
  ├── 1st_origin
  ├── 2nd_process
├── ClassC
```

- **`1st_origin`**：原始数据，通常来自专有来源，统一格式。
- **`2nd_process`**：经过初步处理后的中间结果数据和可用于分析或建模的数据。

### 数据读取

数据读取代码主要在 reader 文件夹中，目前提供的接口功能主要是：

- GRDC、GAGES、CRD等数据集的读取
- 自定义站点数据的读取
- 自建数据集的读取

后续会提供更详细的文档。
