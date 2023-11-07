<!--
 * @Author: Wenyu Ouyang
 * @Date: 2023-10-24 21:30:40
 * @LastEditTime: 2023-11-07 14:30:57
 * @LastEditors: Wenyu Ouyang
 * @Description: Readme for hydro_privatedata
 * @FilePath: \hydro_privatedata\README.md
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

This project is primarily used to handle preliminarily processed internal data on MinIO. Due to the often complex nature of internal data, processing can be quite cumbersome. Hence, the conversion of internal data to MinIO is handled within an internal Gitlab project on a case-by-case basis, without a standardized processing workflow.

The organizational structure of data files on MinIO is as follows (tentative and subject to further refinement):

```
├── datasets-origin
├── basins-origin
│   ├── basins_list
├── reservoirs-origin
│   ├── reservoirs_list
│   ├── res_liaob_biliuhe
│   │   ├── liaob_biliuhe_inflow.csv
├── rivers-origin
├── grids-origin
│   ├── DEM
│   |   ├── xxx (product name)
├── GFS
│   ├── xxxx (year)
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

Here, the `datasets-origin` folder contains data sets; `basins-origin` contains basin data; `reservoirs-origin` stores reservoir data; `rivers-origin` for river data; `grids-origin` for grid data; and `stations-origin` for station data.

Data in the `origin` folders are raw, while the `interim` folders contain data that has been preliminarily processed. Generally speaking, the data in `origin` is the result of processing in Gitlab's case-by-case project, and `interim` is where the data from `origin` is processed according to some specific requirements into the desired format.

**The main function of this project is to read data from the `origin` folders, process it according to certain requirements into the data found in the `interim` folders, and provide some functionalities for synchronizing local folders with MinIO buckets (synchronization is recommended using the terminal tool provided by MinIO).**

## Env settings for coding

```bash
# Create a virtual environment
conda env create -f env-dev.yml
# Activate the virtual environment
conda activate privatedata
```

## Reading data from `origin`

To be completed...

## Processing data from `origin`

To be completed...

## Synchronizing local folders with MinIO buckets

For this part, it is recommended to use the `mirror` command of the MinIO client `mc`.

First, you need to install MinIO's client command-line tool `mc`. This can be done either by downloading the binary directly or by using a package manager.

For Linux, you can install it in the following way (note: this part has not been fully verified):

```bash
wget https://dl.min.io/client/mc/release/linux-amd64/mc
chmod +x mc
sudo mv mc /usr/local/bin/
```

If you're using macOS, you can install it using brew:

```bash
brew install minio/stable/mc
```

For Windows users, you can download the executable file and add it to the system path. The documentation is [here](https://min.io/docs/minio/linux/reference/minio-mc.html?ref=docs). Follow the steps in the documentation:
1. First download the MinIO client program `mc.exe`, and move the `mc.exe` file to the directory where you wish to execute MinIO client commands.
2. Note that you should not double-click `mc.exe` to run it. It needs to be run from the command line. Open the terminal, navigate to the directory, then enter the command: `mc.exe --help`. If the help documentation appears, it means the installation is successful.
3. Add the download directory to the system environment variables so you can use `mc.exe` in any directory. Open the terminal, input `mc --version`, and press Enter to see if the MinIO client version information is output. If the version information is displayed, it means the environment variable configuration is successful.

After installing `mc`, you need to configure it to connect to your MinIO server.

```bash
mc alias set myminio http://minio-server:9000 ACCESS_KEY SECRET_KEY
```

- `myminio` is the alias set, which can be used to reference our MinIO server when executing commands.
- `http://minio-server:9000` is the address and port of the MinIO server.
- `ACCESS_KEY` and `SECRET_KEY` are the access key and secret key for the MinIO server.

Then, you can use the `mc` command to synchronize folders.

```bash
mc mirror /path/to/local/folder myminio/mybucket
```

This command will synchronize the local folder `/path/to/local/folder` to a bucket named `mybucket` in MinIO.

If you want to sync from MinIO to a local folder, you just need to reverse the order:

```bash
mc mirror myminio/mybucket /path/to/local/folder
```

To selectively sync specific buckets or folders, you can use the `--include` and `--exclude` parameters to filter the contents you want to sync. For example:

```bash
# Sync some folders from `mybucket` in MinIO to local
mc mirror --include "/important-data/*" myminio/mybucket /path/to/local/folder
# Exclude certain files or folders you don't need to sync
mc mirror --exclude "/temp/*" myminio/mybucket /path/to/local/folder
```

**It is strongly recommended to create a dedicated folder locally to sync with a MinIO server!!!**
