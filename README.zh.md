<!--
 * @Author: Wenyu Ouyang
 * @Date: 2023-10-25 14:43:12
 * @LastEditTime: 2024-03-23 19:20:11
 * @LastEditors: Wenyu Ouyang
 * @Description: Chinese version README
 * @FilePath: \hydrodata\README.zh.md
 * Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
-->
# hydrodatasource

[![image](https://img.shields.io/pypi/v/hydrodatasource.svg)](https://pypi.python.org/pypi/hydrodatasource)
<!-- [![image](https://img.shields.io/conda/vn/conda-forge/hydrodatasource.svg)](https://anaconda.org/conda-forge/hydrodatasource) -->

-   Free software: MIT license
-   Documentation: <https://hydrodatasource.readthedocs.io/en/latest/>
 
尽管水文领域存在很多流域水文数据集，但是一个很显然的问题是还有很多数据没被整理到公开的数据集中，包括

- 时效性的问题没来得及整理的
- 没被现有数据集考虑到的
- 以及不会被公开的数据。

这些数据占据着相当的比例，尤其是对于中国的水文数据来说。比如，最常用的CAMELS数据集数据到2014年12月，马上也快10年之久了；GRDC径流数据虽然有用，但是也很少被专门做到某个数据集中，GFS、GPM、SMAP等一系列实时近实时的网格数据也很少被做成数据集，更多的是ERA5Land等质量更高的数据被做到数据集中以进行研究；大量不公开的数据自然也无法被拿去构建数据集。

但是这些数据又非常重要，为此，我们构思了这个hydrodatasource仓库，旨在提供一个能统一整理这些数据的方式，使得这些数据能够在以流域为基本单元的科研和生产背景下被更好地利用。关于完整数据集，请关注：[hydrodataset](https://github.com/OuyangWenyu/hydrodataset)

更具体一点来说，这个仓库的目标是提供一个统一的流域水文数据获取、管理和使用路径及方法，使得水文计算尤其是基于人工智能方面的水文模型计算更加方便。

## How many data sources are there

以流域为最终数据描述主体的角度来考虑，目前我们的数据源主要包括以下几类：

| **一级分类** | **二级分类** | **更新频率** | **数据结构** | **具体数据源** |
| --- | --- | --- | --- | --- |
| 基础 | 地理图 | 历史存档 | 矢量 | 流域边界、站点等shapefile |
|  | 高程数据 | 历史存档 | 栅格 | [DEM](https://github.com/DahnJ/Awesome-DEM)|
|  | 属性数据 | 历史存档 | 表格 | HydroATLAS数据集 |
| 气象 | 再分析数据集 | 历史存档、延迟动态 | 栅格 | ERA5Land |
|  | 遥感降水 | 历史存档、近实时动态 | 栅格 | GPM |
|  | 气象模式预报 | 历史存档、实时滚动 | 栅格 | GFS |
|  | AI气象预报 | 实时滚动 | 栅格 | AIFS |
|  | 地面气象站 | 历史存档 | 表格 | NOAA 气象站 |
|  | 地面雨量站 | 历史存档、实时/延迟动态 | 表格 | 非公开雨量站 |
| 水文 | 遥感土壤含水量 | 历史存档、近实时动态 | 栅格 | SMAP |
|  | 墒情站 | 历史存档、实时动态 | 表格 | 非公开的墒情站 |
|  | 地面水文站 | 历史存档 | 表格 | USGS |
|  | 地面水文站 | 历史存档、实时动态 | 表格 | 非公开的水位、流量站 |
|  | 径流数据集 | 历史存档 | 表格 | GRDC |

注：更新频率不完全指实际数据源的更新频率，主要以本仓库的数据更新频率为准。

## What are the main features

在具体使用之前，有必要了解下本仓库的主要特点，这样才能知道怎么使用，希望用户能保持一点耐心。

我们的想法是能让有不同硬件资源的人都能比较方便地使用这个工具。关于硬件资源，这里稍作介绍。由于整个仓库涉及地数据类型很多，数据量也很大，所以作为开发者的我们是构建了一个 MinIO 服务。MinIO 是一个开源的对象存储服务，可以很方便地部署在本地或者云端，我们这里是部署在本地的。这样，我们就可以把数据存储在 MinIO 上，然后通过 MinIO 提供的 API 来读取数据。这样做的好处是，我们能有效地管理大量的数据，和开发统一访问的接口，使得数据的读取更加方便。但是，这样做的缺点是，需要一定的硬件资源，比如硬盘空间、内存等。所以，我们也考虑针对一部分数据提供完全本地文件的交互方式来读取，但是这种方式我们就不会做完全功能的测试覆盖了。

基于上面的基本思路，针对不同的数据，我们的处理方式也有所区别。

- 对于非公开的数据，公开的代码部分主要是考虑提供工具函数，以支持用户自己处理自己的数据，以便后续运行我们提供的开源模型。当然，开发者自己内部会提供数据的读取服务。
- 对于公开的数据，我们会提供一些数据下载、格式转换和读取的代码，以支持用户在自己本地上操作数据。

接下来，我们就按这两部分来展开。

### For non-public data

非公开的数据主要就是地面站点的数据，所以我们就针对这部分数据，提供一些数据转换格式的工具，我们会定义一个用户需要准备的数据格式，然后后续的部分就直接调用工具即可。总的来说，我们会希望用户按照一定的表格格式准备自己的数据，然后我们会处理成 netcdf 格式的数据，以便后续的模型读取。至于具体要准备的数据格式，我们提供了一个data_checker函数来检查数据格式，用户可以通过这个函数来了解数据的具体格式。后续我们也会补充一个文档来详细说明数据的具体格式。

### For public data

公开的数据主要是一些已经被整理成数据集的数据，我们会提供一些数据下载、格式转换和读取的代码，以支持用户在自己本地上操作数据。这部分数据主要是一些已经被整理成数据集的数据，比如 CAMELS、GRDC、ERA5Land 等。

但是，如前所述，我们不会提供针对本地文件的完整测试覆盖，我们主要在MinIO上测试相关代码。

## How to use

### Installation

We recommend installing the package via pip:

```bash
pip install hydrodatasource
```

### Usage

我们约定的数据文件组织结构第一集目录是这样的：
    
```
├── datasets-origin
├── datasets-interim
├── basins-origin
├── basins-interim
├── reservoirs-origin
├── reservoirs-interim
├── grids-origin
├── grids-interim
├── stations-origin
├── stations-interim
```

其中，datasets-origin文件夹存放的是数据集，basins-origin文件夹存放的是流域数据，reservoirs-origin文件夹存放的是水库数据，rivers-origin文件夹存放的是河流数据，grids-origin文件夹存放的是格点数据，stations-origin文件夹存放的是站点数据。

origin文件夹中的数据是原始数据，interim文件夹中的数据是经过初步处理的数据，基本上来说，origin中的数据就是前期在gitlab的一事一议项目中处理之后的数据结果，interim就是这里要把origin的数据根据一项什么具体需求处理成什么格式后得到的数据。

这样的分类能完全覆盖表格中的数据类型。

对于非公开站点数据：

1. 首先，用户需要准备好自己的数据，数据格式要求是表格格式，执行下面的命令了解数据的具体格式：
    ```python
    from hydrodatasource import station
    station.get_station_format()
    ```
2. 把文件放到文件夹 stations-origin 中，具体的上级绝对路径请在自己电脑用户文件夹下面的 hydro_settings.yml文件中配置。