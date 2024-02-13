<!--
 * @Author: Wenyu Ouyang
 * @Date: 2023-10-26 08:41:30
 * @LastEditTime: 2024-02-13 17:58:25
 * @LastEditors: Wenyu Ouyang
 * @Description: 
 * @FilePath: \hydrodata\docs\installation.md
 * Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
-->
# Installation

## Stable release

To install hydroommand in your terminal:

```
pip install hydro
```

This is the preferred method to install hydrodata, as it will always install the most recent stable release.

If you don't have [pip](https://pip.pypa.io) installed, this [Python installation guide](http://docs.python-guide.org/en/latest/starting/installation/) can guide you through the process.

## From sources

To install hydrodata from sources, run this command in your terminal:

```
pip install git+https://github.com/WenyuOuyang/hydrodata
```

# 安装与配置

## 安装

在命令行运行：

```shell
pip install hydrodata
```

## 配置

首次使用hydrodata时，如：

```python
from hydrodata.data.minio import GPM
```

会提示输入`access_key`和`secret_key`。由于目前hydrodata部分功能仅限于团队内部使用，所以使用前需要向管理员申请[minio](http://minio.waterism.com:9090/)账号。

1. 进入[minio网页客户端](http://minio.waterism.com:9090/)
2. 点击左侧导航栏`Identity`进入`Service Accounts`
3. 在右侧主界面点击`Create Service Accounts`

![获取key](./images/account.png)