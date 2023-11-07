<!--
 * @Author: Wenyu Ouyang
 * @Date: 2023-10-25 14:43:12
 * @LastEditTime: 2023-11-07 14:28:23
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

**本项目的功能主要是读取origin中的数据，然后根据一些具体需求，处理成interim中的数据，并提供一些本地文件夹和minio桶同步的功能（同步推荐实用minio提供的终端工具）。**

## 读取origin数据

待完成……

## 处理origin数据

待完成……

## 同步本地文件夹和minio桶

这部分推荐使用 MinIO 客户端 mc 的 mirror 命令

首先，需要安装 MinIO 的客户端命令行工具 mc。这可以通过直接下载二进制文件或者使用包管理器来完成。

如果使用 Linux，可以通过以下方式安装（注：这部分没完全验证）：

```bash
wget https://dl.min.io/client/mc/release/linux-amd64/mc
chmod +x mc
sudo mv mc /usr/local/bin/
```

如果使用macOS，可以使用brew来安装：

```bash
brew install minio/stable/mc
```

对于Windows用户，则可以下载可执行文件，将其添加到系统路径中，文档在[这里](https://min.io/docs/minio/linux/reference/minio-mc.html?ref=docs)，按照文档中的步骤操作即可：

1. 先下载minio客户端程序mc.exe，将mc.exe文件移动到希望执行MinIO客户端命令的目录
2. 注意不要双击 mc.exe 运行，需要在命令行运行，打开终端，进入到目录，然后输入命令：`mc.exe --help`，如果出现了帮助文档，说明安装成功了。
3. 将下载目录添加到系统环境变量中，这样就可以在任意目录下使用mc.exe了，打开终端，输入 `mc --version` 并按Enter键，查看是否输出了MinIO客户端的版本信息，如果输出了版本信息，说明环境变量配置成功了。

在安装了 mc 之后，您需要配置它以连接到您的 MinIO 服务器。

```bash
mc alias set myminio http://minio-server:9000 ACCESS_KEY SECRET_KEY
```

- myminio 是设置的别名，可以在执行命令时使用它来引用我们的MinIO服务器。
- http://minio-server:9000 是MinIO服务器的地址和端口。
- ACCESS_KEY 和 SECRET_KEY 是MinIO服务器的访问密钥和秘密密钥。

然后，就可以使用mc命令来同步文件夹了。

```bash
mc mirror /path/to/local/folder myminio/mybucket
```

这个命令将会同步本地文件夹/path/to/local/folder到MinIO中名为mybucket的桶。

如果你想要从MinIO同步到本地，你只需要调换顺序：

```bash
mc mirror myminio/mybucket /path/to/local/folder
```

要选择性地同步特定的桶或文件夹，你可以使用--include和--exclude参数来过滤要同步的内容。例如：

```bash
# 同步mybucket中的某些文件夹到本地
mc mirror --include "/important-data/*" myminio/mybucket /path/to/local/folder
# 排除某些不需要同步的文件或文件夹
mc mirror --exclude "/temp/*" myminio/mybucket /path/to/local/folder
```

**强烈建议在你本地建一个专门用来和一个minio服务器同步的文件夹！！！**
