<!--
 * @Author: Wenyu Ouyang
 * @Date: 2023-10-24 21:30:40
 * @LastEditTime: 2024-06-25 11:42:16
 * @LastEditors: Wenyu Ouyang
 * @Description: Readme for hydrodatasource
 * @FilePath: \hydrodatasource\README.md
 * Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
-->
# hydrodatasource


[![image](https://img.shields.io/pypi/v/hydrodatasource.svg)](https://pypi.python.org/pypi/hydrodatasource)
[![image](https://img.shields.io/conda/vn/conda-forge/hydrodatasource.svg)](https://anaconda.org/conda-forge/hydrodatasource)

[![image](https://pyup.io/repos/github/iHeadWater/hydrodatasource/shield.svg)](https://pyup.io/repos/github/iHeadWater/hydrodatasource)

-   Free software: BSD license
-   Documentation: https://WenyuOuyang.github.io/hydrodatasource

📜 [中文文档](README.zh.md)


Although there are many hydrological datasets for various watersheds, a noticeable issue is that many data sources remain unorganized and are not part of public datasets. This includes data that hasn't been organized due to its recency, data not considered by existing datasets, and data that will not be made public. These data sources represent a significant portion of available data. For example, the commonly used CAMELS dataset only includes data up to December 2014, almost ten years ago; GRDC runoff data, while useful, is rarely included in specific datasets. Real-time and near-real-time gridded data such as GFS, GPM, SMAP, etc., are infrequently compiled into datasets, with more emphasis on higher quality data like ERA5Land being used for research. A large portion of hydrological data in China is not public, and thus cannot be used to construct datasets.

To address this, we conceived the hydrodatasource repository, aiming to provide a unified way of organizing these data sources for better utilization in scientific research and production, especially within the context of watersheds. For information on currently available public datasets, please visit: [hydrodataset](https://github.com/OuyangWenyu/hydrodataset).

To be more specific, the goal of this repository is to provide a unified pathway and method for watershed hydrological data management and usage, making hydrological model calculations, especially those based on artificial intelligence, more convenient.

Regarding the part about data acquisition, since it involves a process with significant manual and semi-automatic intervention, we have placed these contents in a separate repository: [HydroDataCompiler](https://github.com/iHeadWater/HydroDataCompiler). Once it is relatively perfected, we will open source this repository.

## How many data sources are there

Considering watersheds as the primary focus of data description, our data sources mainly include:

| **Primary Category** | **Secondary Category** | **Update Frequency** | **Data Structure** | **Specific Data Source** |
| --- | --- | --- | --- | --- |
| Baseline | Geographic Maps | Historical Archive | Vector | Watershed boundaries, site locations, and other shapefiles |
|  | Elevation Data | Historical Archive | Raster | [DEM](https://github.com/DahnJ/Awesome-DEM) |
|  | Attribute Data | Historical Archive | Tabular | HydroATLAS dataset |
| Meteorological | Reanalysis Data Sets | Historical Archive, Delayed Dynamic | Raster | ERA5Land |
|  | Remote Sensing Precipitation | Historical Archive, Near Real-Time Dynamic | Raster | GPM |
|  | Weather Model Forecasts | Historical Archive, Real-Time Rolling | Raster | GFS |
|  | AI Weather Forecasts | Real-Time Rolling | Raster | AIFS |
|  | Ground Weather Stations | Historical Archive | Tabular | NOAA weather stations |
|  | Ground Rainfall Stations | Historical Archive, Real-Time/Delayed Dynamic | Tabular | Non-public rainfall stations |
| Hydrology | Remote Sensing Soil Moisture | Historical Archive, Near Real-Time Dynamic | Raster | SMAP |
|  | Soil Moisture Stations | Historical Archive, Real-Time Dynamic | Tabular | Non-public soil moisture stations |
|  | Ground Hydrological Stations | Historical Archive | Tabular | USGS |
|  | Ground Hydrological Stations | Historical Archive, Real-Time Dynamic | Tabular | Non-public water level and flow stations |
|  | Runoff Data Sets | Historical Archive | Tabular | GRDC |

Note: The update frequency primarily refers to the frequency of updates in this repository, not necessarily the actual data source's update frequency.

## What are the main features

Before using it, it is essential to understand the main features of this repository, as this will guide its use.

Our goal is to make this tool accessible to users with varying hardware resources. To elaborate on hardware resources: due to the extensive variety and volume of data involved, we have set up a MinIO service. MinIO is an open-source object storage service, which can be conveniently deployed locally or in the cloud; in our case, it's deployed locally. Thus, data is stored on MinIO and accessed via its API. This approach allows effective data management and the development of a unified access interface, simplifying data retrieval. However, it does require specific hardware resources, like disk space and memory. Therefore, we also offer a fully local file interaction mode for a portion of the data, although this mode won't be covered by complete functional testing.

Based on this approach, we handle different types of data differently:

For non-public data, we mainly provide utility functions in the public code to assist users in processing their data, facilitating the use of our open-source models. Of course, developers internally provide data retrieval services for their own data.
For public data, we offer code for data download, format conversion, and reading, supporting users in handling data on their local systems.
Now, let's expand on these two parts.

### For non-public data

The non-public data primarily involves ground station data. We provide tools for data format conversion for these data types. We define a data format that users need to prepare, and the subsequent process involves using these tools directly. In general, we expect users to prepare their data in a specific tabular format, which we will then convert into netCDF format for model reading. As for the exact format to prepare, we provide a data_checker function to verify the data format. Users can use this function to understand the specifics. We will also add a document detailing the specific format, which is yet to be completed.

### For public data

The public data mainly consists of those already organized into datasets. We provide code for data download, format conversion, and reading to support users in operating data on their local systems. These datasets include, but are not limited to, CAMELS, GRDC, ERA5Land, etc.

However, as previously mentioned, we do not provide complete test coverage for local files. Our primary testing is conducted on MinIO.

## How to use

### Installation

We recommend installing the package via pip:

```bash
pip install hydrodatasource
```

### Usage

Our agreed data file organization structure at the primary level looks like this:

```dir
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

Here, datasets-origin contains the datasets, basins-origin contains watershed data, reservoirs-origin stores reservoir data, rivers-origin holds river data, grids-origin includes gridded data, and stations-origin has station data.

Data in the origin folders is raw data, while the interim folders contain data that has undergone preliminary processing. Essentially, the data in origin is the result of initial processing in GitLab's One Thing One Vote project, and interim is where origin data is processed into a specific format based on a particular requirement.

This categorization fully covers the types of data listed in the table.

For non-public station data:

First, users need to prepare their data in a tabular format. To understand the specific format required, execute the following command:

```python
from hydrodatasource import station
station.get_station_format()
```

Place the files in the stations-origin folder. For the specific parent absolute path, please configure it in the hydro_settings.yml file in your computer's user folder.