<!--
 * @Author: Wenyu Ouyang
 * @Date: 2023-10-24 21:30:40
 * @LastEditTime: 2023-10-25 15:21:15
 * @LastEditors: Wenyu Ouyang
 * @Description: Readme for 
 * @FilePath: /hydro_privatedata/README.md
 * Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
-->
# hydro_privatedata


[![image](https://img.shields.io/pypi/v/hydro_privatedata.svg)](https://pypi.python.org/pypi/hydro_privatedata)
[![image](https://img.shields.io/conda/vn/conda-forge/hydro_privatedata.svg)](https://anaconda.org/conda-forge/hydro_privatedata)

[![image](https://pyup.io/repos/github/WenyuOuyang/hydro_privatedata/shield.svg)](https://pyup.io/repos/github/WenyuOuyang/hydro_privatedata)


**A python project to deal with internal datasources**


-   Free software: BSD license
-   Documentation: https://WenyuOuyang.github.io/hydro_privatedata

📜 [中文文档](README.zh.md)

## Introduction

Data processing for various databases, sheet files, etc. in our own servers

Typically, we have the following data sources:

- Databases from various projects
- Sheet files from various projects

To process these data, we have the following steps:

1. For databases, we creata a central database and connect it to the databases, then we extract/transform/load the data to the central database.
2. For sheet files, we save all original files in one folder, then we transform them to tidy data format. All files are saved in one directory in our central server, and some of them are saved in the central database if necessary.

## Sheet files

### Data format

There is no standard data format for sheet files, various original data formats are used in different projects. Hence, we will gradually transform them to tidy data format. When we meet one strange format, we will record it here and write a script to transform it to tidy data format.

The tidy data format is from [R for Data Science](https://r4ds.had.co.nz/tidy-data.html). The basic principles are:

1. Each variable forms a column.
2. Each observation forms a row.
3. Each type of observational unit forms a table.

Typically, it has a same structure as a database table. The difference is that the tidy data format is saved in a sheet file, while the database table is saved in a database.

## Databases

Each project has its own database, and we have a central database to connect all databases. The central database is used to extract/transform/load data from various databases.

## Env settings

```bash
# Create a virtual environment
conda env create -f env-dev.yml
# Activate the virtual environment
conda activate privatedata
```
