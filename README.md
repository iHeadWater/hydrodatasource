<!--
 * @Author: Wenyu Ouyang
 * @Date: 2023-10-24 21:30:40
 * @LastEditTime: 2025-01-09 17:16:58
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


## Overview

Although numerous public watershed hydrological datasets are available, there are still challenges in this field:

- Many datasets are not updated or included in subsequent versions.
- Some datasets remain uncovered by existing collections.
- Non-public datasets cannot be openly shared.

To address these issues, **hydrodatasource** provides a framework to organize and manage these datasets, making them more efficient for use in watershed-based research and production scenarios.

This repository complements [hydrodataset](https://github.com/OuyangWenyu/hydrodataset), which focuses on public datasets. In contrast, **hydrodatasource** integrates a broader range of data resources, including non-public and custom datasets.

## Data Classification and Sources

**hydrodatasource** processes data that primarily falls into three categories:

### Category A Data (Public Data)

These datasets are organized and managed according to predefined formats, including:

1. **GIS Datasets**: Geographic vector data, such as watershed boundaries and station shapefiles.
2. **Gridded Datasets**: Includes datasets like ERA5Land, GPM, and AIFS, which are stored in a MinIO database.

### Category B Data (Non-Public or Industry Data)

These datasets are often proprietary or confidential and require specific tools for formatting and integration, including:

1. **Custom Station Data**: User-prepared station data formatted into NetCDF for seamless model usage.
2. **Industry Data**: Professionally integrated and formatted datasets.

### Custom Hydrological Datasets

Based on Category A and B data, custom hydrological datasets are created, adhering to predefined standard formats for specific research needs.

## Features and Highlights

### Unified Data Management

**hydrodatasource** provides standardized methods for:

- Structuring datasets according to predefined conventions.
- Integrating various data sources into a unified framework.
- Supporting data access and processing for hydrological modeling.

### Compatibility with Local and Cloud Resources

- **Public Data**: Supports data format conversion and local file operations.
- **Non-Public Data**: Provides tools to format and integrate user-prepared data.
- **MinIO Integration**: Efficient management of large-scale gridded data via API.

### Modular Design

The repository structure supports diverse workflows, including:

1. **Category A GIS Data**: Tools to organize and access GIS datasets.
2. **Category A Gridded Data**: Large-scale grid data management via MinIO.
3. **Category B Data**: Custom tools to clean and process station, reservoir, and basin time-series data.
4. **Custom Hydrological Datasets**: Support for predefined dataset formats.

### Other Interactions

**hydrodatasource** interacts with the following components:

- [**hydrodataset**](https://github.com/OuyangWenyu/hydrodataset): Provides necessary support for accessing public datasets.
- [**HydroDataCompiler**](https://github.com/iHeadWater/HydroDataCompiler): Supports semi-automated processing of non-public and custom data (currently not public).
- [**MinIO Database**](http://10.48.0.86:9001/): Efficient storage and management of Category A gridded data (currently accessible only within the internal network).

## Installation

Install the package via pip:

```bash
pip install hydrodatasource
```

Note: The project is still in the early stages of development, so development mode is recommended.

## Usage

### Data Organization

The repository adopts the following directory structure for organizing data:

```
├── datasets-origin          # Public hydrological datasets
├── datasets-interim         # Custom hydrological datasets
├── gis-origin               # Public GIS datasets
├── grids-origin             # Gridded datasets
├── stations-origin          # Category B station data (raw)
├── stations-interim         # Category B station data (processed)
├── reservoirs-origin        # Category B reservoir data (raw)
├── reservoirs-interim       # Category B reservoir data (processed)
├── basins-origin            # Category B basin data (raw)
├── basins-interim           # Category B basin data (processed)
```

- **`origin`**: Raw data, often from proprietary sources, in unified formats.
- **`interim`**: Preprocessed data ready for analysis or modeling.

### For Category A Data

1. **Public GIS Data**:
   - Store vector files in the `gis-origin` folder, such as watershed boundaries and station shapefiles.
   - Process the data using:
     ```python
     from hydrodatasource import gis
     gis.process_gis_data(input_path="gis-origin", output_path="gis-interim")
     ```

2. **Gridded Datasets**:
   - Store raw grid data in `grids-origin`, such as ERA5Land and GPM.
   - Use MinIO API to download or manage data stored in the database:
     ```python
     from hydrodatasource import grid
     grid.download_from_minio(dataset_name="ERA5Land", save_path="grids-interim")
     ```

### For Category B Data

1. **Station Data**:
   - Store raw data in the `stations-origin` folder and processed data in `stations-interim`.
   - Check the standard format for station data:
     ```python
     from hydrodatasource import station
     station.get_station_format()
     ```
   - Format and process the data:
     ```python
     station.process_station_data(input_path="stations-origin", output_path="stations-interim")
     ```

2. **Reservoir Data**:
   - Store raw reservoir data in the `reservoirs-origin` folder and cleaned data in `reservoirs-interim`.
   - Specific tools are provided for integration and formatting.

3. **Basin Data**:
   - Store raw basin data in the `basins-origin` folder and processed data in `basins-interim`.
   - These datasets typically include attributes and spatial information supporting hydrological modeling.

### Custom Hydrological Datasets

Custom datasets are stored in the `datasets-interim` folder. They are organized according to predefined standard formats to facilitate integration and subsequent model use.

### MinIO Database and Data Storage

The MinIO database is primarily used for storing and managing large-scale gridded data (Category A data), such as ERA5Land and other dynamic datasets:

- Configure MinIO access in the `hydro_settings.yml` file.
- Upload or download data:
  ```python
  from hydrodatasource import minio
  minio.upload_to_minio(local_path="grids-interim/ERA5Land", dataset_name="ERA5Land")
  ```

## Conclusion

**hydrodatasource** bridges diverse hydrological datasets and advanced modeling needs by providing standardized workflows and modular tools. This ensures efficient data management and integration, supporting both research and operational applications.