<!--
 * @Author: Wenyu Ouyang
 * @Date: 2023-10-25 14:43:12
 * @LastEditTime: 2024-02-12 12:59:44
 * @LastEditors: Wenyu Ouyang
 * @Description: Chinese version README
 * @FilePath: \hydrodata\README.zh.md
 * Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
-->
# hydro_privatedata

## 简介

本项目主要用来处理minio上初步处理的内部数据，因为内部数据通常形式比较繁杂，因此处理起来会很麻烦，所以内部数据转换到minio上的过程在内部Gitlab项目中进行处理，一事一议，就不再（也基本没法）制作标准化的处理流程了。

minio上的数据文件组织结构是这样的（暂定，还需进一步完善）：
    
```
├── datasets-origin
├── basins-origin
│   ├── basins_list
├── reservoirs-origin
│   ├── reservoirs_list
│   ├── res_liaob_biliuhe
│   │   ├── liaob_biliuhe_inflow.csv
├── rivers-origin
├── grids-origin
│   ├── DEM
│   |   ├── xxx (产品名)
├── GFS
│   ├── xxxx (年份)
├── stations-origin
│   ├── stations_list
│   │  ├── pp.csv
│   ├── pp_stations
│   │  ├── pp_1_01013500.csv
├── basins-interim
│   ├── 1_01013500_datatype_process
├── grids-interim
├── stations-interim
```

其中，datasets-origin文件夹存放的是数据集，basins-origin文件夹存放的是流域数据，reservoirs-origin文件夹存放的是水库数据，rivers-origin文件夹存放的是河流数据，grids-origin文件夹存放的是格点数据，stations-origin文件夹存放的是站点数据。

origin文件夹中的数据是原始数据，interim文件夹中的数据是经过初步处理的数据，基本上来说，origin中的数据就是前期在gitlab的一事一议项目中处理之后的数据结果，interim就是这里要把origin的数据根据一项什么具体需求处理成什么格式后得到的数据。

**本项目的功能主要是读取origin中的数据，然后根据一些具体需求，处理成interim中的数据，并提供一些本地文件夹和minio桶同步的功能（同步推荐实用minio提供的终端工具）。还有一个预期的功能是在读取本地或者minio的数据时让用户没有感觉，只要通过简单设定，就能从minio或者本地读取同样的数据**

## 读取origin数据

待完成……

## 处理origin数据

待完成……

## 同步本地文件夹和minio桶

这部分推荐使用 MinIO 客户端 mc 的 mirror 命令，具体的技术细节可以参考内部的技术文档。

## 构建兼容本地和minio数据的读取器

如果要兼容之前所有的数据，那么本质上就是需要把原来所有的数据读取代码从只能读本地的重构到能兼容minio和本地的，因为这个之前单纯就是科研，所以没太考虑，那么就导致如果具体执行，这个工作量是很大的。

所以我们要采取一种就是有优先级重要性的建设方式，像成熟的数据集和我们自己内部构建的流域集总式（流域均值）的数据集，这种数据量也都不大，可以先不管了，就还维持原来的读取方式，但是对于一些大数据量的数据集，比如格点数据，就需要重点考虑建设这个读取器了。

基本的构建思路是这样的：

1. 中心化数据读取逻辑：创建一个统一的数据读取函数，所有需要读取数据的地方都调用这个函数。这样，只需要在一个地方修改读取数据的逻辑，而不是在代码的多个地方。当然具体实现还得结合具体的情况，但是要尽可能地把数据读取的功能统一到一个module下面，然后通过不同的函数来实现各类具体的读取逻辑。
2. 使用配置来切换数据源：使用配置和一点点设计模式，例如策略模式等，来实现数据源的切换。
3. 逐步重构：代码涉及读取数据的环节可能很多，所以就一点点地修改测试，先把数据读取比较密集频繁的地方，比如torch dataset类里面的内容重构了，再统一检查其他地方，逐步完善。
   
# hydro-opendata


[![image](https://img.shields.io/pypi/v/hydro-opendata.svg)](https://pypi.python.org/pypi/hydro-opendata)
<!-- [![image](https://img.shields.io/conda/vn/conda-forge/hydro-opendata.svg)](https://anaconda.org/conda-forge/hydro-opendata) -->


**可用于水文学科学计算的开放数据的获取、管理和使用路径及方法。**


-   Free software: MIT license
-   Documentation: <https://hydro-opendata.readthedocs.io/en/latest/>
 
## 背景

在人工智能的大背景下，数据驱动的水文模型已得到广泛研究和应用。同时，得益于遥测技术发展和数据开放共享，获取数据变得容易且选择更多了。对于研究者而言，需要什么数据？能获取什么数据？从哪下载？怎么读取？如何处理？等一系列问题尤为重要，也是源平台数据中心建设需要解决的问题。

本仓库主要基于外部开放数据，梳理数据类别和数据清单，构建能够实现数据“下载-存储-处理-读写-可视化”的数据流及其技术栈。

## 总体方案

![数据框架图](images/framework.png)

## 主要数据源

从现有认识来看，可用于水文建模的外部数据包括但不限于以下几类：

| **一级分类** | **二级分类** | **更新频率** | **数据结构** | **示例** |
| --- | --- | --- | --- | --- |
| 基础地理 | 水文要素 | 静态 | 矢量 | 流域边界、站点 |
|  | 地形地貌 | 静态 | 栅格 | [DEM](https://github.com/DahnJ/Awesome-DEM)、流向、土地利用 |
| 天气气象 | 再分析 | 动态 | 栅格 | ERA5 |
|  | 近实时 | 动态 | 栅格 | GPM |
|  | 预测 | 滚动 | 栅格 | GFS |
| 图像影像 | 卫星遥感 | 动态 | 栅格 | Landsat、Sentinel、MODIS |
|  | 街景图片 | 静态 | 多媒体 |  |
|  | 监控视频 | 动态 | 多媒体 |  |
|  | 无人机视频 | 动态 | 多媒体 |  |
| 众包数据 | POI | 静态 | 矢量 | 百度地图 |
|  | 社交网络 | 动态 | 多媒体 | 微博 |
| 水文数据 | 河流流量数据 | 动态 | 表格 | GRDC |

从数据更新频率上来看，分为静态数据和动态数据。

从数据结构上看，分为矢量、栅格和多媒体数据等非结构化数据。

## 结构及功能框架

![代码仓](images/repos.jpg)

### wis-stac

数据清单及其元数据，根据AOI返回数据列表。

### wis-downloader

从外部数据源下载数据。根据数据源不同，下载方法不尽相同，主要包括：

- 通过集成官方提供的api，如[bmi_era5](https://github.com/gantian127/bmi_era5)
- 通过获取数据的下载链接，如[Herbie](https://github.com/blaylockbk/Herbie)、[MultiEarth](https://github.com/bair-climate-initiative/multiearth)、[Satpy](https://github.com/pytroll/satpy)，大部分云数据平台如Microsoft、AWS等数据组织的方式大多为[stac](https://github.com/radiantearth/stac-spec)

### wis-processor

对数据进行预处理，如流域平局、提取特征值等。

使用[kerchunk](https://fsspec.github.io/kerchunk/)将不同格式数据转换成[zarr](https://zarr.readthedocs.io/en/stable/)格式存储到[MinIO](http://minio.waterism.com:9090/)服务器中，实现数据的跨文件读取，提高数据读取效率。

### wis-s3api

数据在MinIO中经过上述写块处理后，即可跨文件读取。只需要提供数据的类别、时间范围和空间范围等参数即可读取数据。

对于遥感影像数据，数据量大且多，无法逐一下载后读取。可以采用[stac+stackstac](./data_api/examples/RSImages.ipynb)直接将Sentinel或Landsat数据读入到xarray的dataset中。


### wis-gistools

集成一些常用的GIS工具，如克里金插值、泰森多边形等。

- 克里金插值
    - [PyKrige](https://github.com/GeoStat-Framework/PyKrige)
- 泰森多边形
    - [WhiteboxTools.VoronoiDiagram](https://whiteboxgeo.com/manual/wbt_book/available_tools/gis_analysis.html?highlight=voro#voronoidiagram)
    - [scipy.spatial.Voronoi](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Voronoi.html)
- 流域划分
    - [Rapid Watershed Delineation using an Automatic Outlet Relocation Algorithm](https://github.com/xiejx5/watershed_delineation)
    - [High-performance watershed delineation algorithm for GPU using CUDA and OpenMP](https://github.com/bkotyra/watershed_delineation_gpu)
- 流域平均
    - [plotting and creation of masks of spatial regions](https://github.com/regionmask/regionmask)

## 可视化

在Jupyter平台中使用[leafmap](https://github.com/giswqs/leafmap)展示地理空间数据。

## 其它

- [hydro-GIS资源目录](./resources/README.md)
