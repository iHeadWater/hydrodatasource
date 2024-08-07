<!--
 * @Author: Wenyu Ouyang
 * @Date: 2024-03-28 09:39:58
 * @LastEditTime: 2024-05-20 20:46:15
 * @LastEditors: Wenyu Ouyang
 * @Description: 
 * @FilePath: \hydrodatasource\docs\changelog.md
 * Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
-->
# 更新日志

## 20231011更新：
**更新内容：**
1. 增加了gpm1d数据；
2. 通过catalog的datasets属性可获取数据集范围。

## 20230905更新：
**更新内容：**
1. 更新minio后台gpm数据，加入了camels范围的gpm数据；
2. 原stac包下各元素整合为catalog，原s3api下各元素整合为reader；
3. 整合了catalog和reader形成data包。

## 20230825更新：
**更新内容：**
1. 合并原wis-stac、wis-s3api、wis-downloader到hydrodatasource。

## 20230824更新：
**更新内容：**
1. 更新wis-s3api/gfs.py中open_datasat方法及其它相关代码；
2. 更新了后台gfs降雨数据实时`下载-上传-切分-访问`代码。

## 20230820更新：
**更新内容：**
1. 更新wis-s3api/gpm.py中open_datasat方法及其它相关代码；
2. 更新了后台gpm数据实时`下载-上传-切分-访问`代码。

**存在问题**
1. gpm从2023年7月2日开始更新了数据格式，更新前后数据的时间格式不一致；
2. 时间范围较大的话读取速度慢；
3. 后台实时下载可能出现错误（目前每个小时运行一次后台任务，正常每次下载2个数据）


