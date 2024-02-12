<!--
 * @Author: Wenyu Ouyang
 * @Date: 2023-10-24 21:30:40
 * @LastEditTime: 2024-02-12 12:59:28
 * @LastEditors: Wenyu Ouyang
 * @Description: Readme for hydro_privatedata
 * @FilePath: \hydrodata\README.md
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

# hydro-opendata

[![image](https://img.shields.io/pypi/v/hydro-opendata.svg)](https://pypi.python.org/pypi/hydro-opendata)


📜 [中文文档](README.zh.md)

**Methods and paths for obtaining, managing, and utilizing open data for hydrological scientific computations.**

- Free software: MIT license
- Documentation: <https://hydro-opendata.readthedocs.io/en/latest/>
- 
## Background

In the era of artificial intelligence, data-driven hydrological models have been extensively researched and applied. With the advancements in remote sensing technologies and the trend towards open data sharing, accessing data has become more straightforward with a plethora of options. For researchers, questions like what data is required, what data can be accessed, where to download it, how to read it, and how to process it, are crucial. This repository aims to address these concerns.

This repository primarily focuses on external open data, categorizing data types, and creating a list. It aims to build a data flow and its tech stack that can seamlessly "download-store-process-read-write-visualize" the data.

## Overall Solution

![Data Framework](images/framework.png)

## Main Data Sources

From our current understanding, the external data suitable for hydrological modeling includes but is not limited to:

| **Primary Category** | **Secondary Category** | **Update Frequency** | **Data Structure** | **Example** |
| --- | --- | --- | --- | --- |
| Basic Geography | Hydrological Elements | Static | Vector | Watershed boundary, site |
|  | Terrain | Static | Raster | [DEM](https://github.com/DahnJ/Awesome-DEM), flow direction, land use |
| Weather & Meteorology | Reanalysis | Dynamic | Raster | ERA5 |
|  | Near Real-Time | Dynamic | Raster | GPM |
|  | Forecast | Rolling | Raster | GFS |
| Imagery | Satellite Remote Sensing | Dynamic | Raster | Landsat, Sentinel, MODIS |
|  | Street View Images | Static | Multimedia |  |
|  | Surveillance Videos | Dynamic | Multimedia |  |
|  | Drone Footage | Dynamic | Multimedia |  |
| Crowdsourced Data | POI | Static | Vector | Baidu Map |
|  | Social Networks | Dynamic | Multimedia | Weibo |
| Hydrological Data | River Flow Data | Dynamic | Tabular | GRDC |

Data can be categorized based on their update frequency into static and dynamic data.

From a structural perspective, data can be classified into vector, raster, and multimedia (unstructured data).

## Structure and Functional Framework

![Code Repository](images/repos.jpg)

### wis-stac

Data inventory and its metadata. Returns a data list based on AOI.

### wis-downloader

Downloads data from external sources. Depending on the data source, the download methods may vary, including:

- Integration with official APIs, e.g., [bmi_era5](https://github.com/gantian127/bmi_era5)
- Retrieving data download links, e.g., [Herbie](https://github.com/blaylockbk/Herbie), [MultiEarth](https://github.com/bair-climate-initiative/multiearth), [Satpy](https://github.com/pytroll/satpy). Most cloud data platforms like Microsoft, AWS, etc., organize data mostly as [stac](https://github.com/radiantearth/stac-spec).

### wis-processor

Preprocesses the data, such as watershed averaging, feature extraction, etc.

Uses [kerchunk](https://fsspec.github.io/kerchunk/) to convert different format data to [zarr](https://zarr.readthedocs.io/en/stable/) format and stores it in [MinIO](http://minio.waterism.com:9090/) server. This enables cross-file reading and enhances data reading efficiency.

### wis-s3api

After data processing in MinIO, it supports cross-file reading. Just provide data type, time range, and spatial range parameters to fetch the data.

For remote sensing imagery, due to the vast amount of data, it's not feasible to download and read each file. One can use [stac+stackstac](./data_api/examples/RSImages.ipynb) to directly read Sentinel or Landsat data into an xarray dataset.

### wis-gistools

Integrates commonly used GIS tools, such as Kriging interpolation, Thiessen polygons, etc.

- Kriging interpolation: [PyKrige](https://github.com/GeoStat-Framework/PyKrige)
- Thiessen polygon: [WhiteboxTools.VoronoiDiagram](https://whiteboxgeo.com/manual/wbt_book/available_tools/gis_analysis.html?highlight=voro#voronoidiagram), [scipy.spatial.Voronoi](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Voronoi.html)
- Watershed delineation: [Rapid Watershed Delineation using an Automatic Outlet Relocation Algorithm](https://github.com/xiejx5/watershed_delineation), [High-performance watershed delineation algorithm for GPU using CUDA and OpenMP](https://github.com/bkotyra/watershed_delineation_gpu)
- Watershed averaging: [plotting and creation of masks of spatial regions](https://github.com/regionmask/regionmask)

## Visualization

Use [leafmap](https://github.com/giswqs/leafmap) to display geospatial data within the Jupyter platform.

## Others

- [hydro-GIS resource directory](./resources/README.md)
