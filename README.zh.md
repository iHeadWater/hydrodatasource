<!--
 * @Author: Wenyu Ouyang
 * @Date: 2023-10-25 14:43:12
 * @LastEditTime: 2023-12-21 15:52:08
 * @LastEditors: Wenyu Ouyang
 * @Description: Chinese version README
 * @FilePath: \hydro_privatedata\README.zh.md
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
   