<!--
 * @Author: Wenyu Ouyang
 * @Date: 2023-10-24 21:30:40
 * @LastEditTime: 2025-06-09 17:01:10
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

- Many datasets are not updated or included in subsequent versions after initial organization.
- Some datasets remain uncovered by existing collections.
- Non-public datasets cannot be directly shared.

To address these issues, **hydrodatasource** provides a framework to organize and manage these datasets, making them more efficient for use in watershed-based research and production scenarios.

This repository works in conjunction with [hydrodataset](https://github.com/OuyangWenyu/hydrodataset), which focuses on public datasets for hydrological modeling. In contrast, **hydrodatasource** integrates a broader range of data resources, including non-public and custom datasets.

## Data Classification and Sources

**hydrodatasource** processes data that primarily falls into three categories:

### Category A Data (Public Data)

These are typically publicly available hydrological datasets from academic papers, currently including:

- GAGES dataset
- GRDC dataset  
- CRD and other reservoir datasets

### Category B Data (Non-Public Data)

These datasets are often proprietary or confidential and require specific tools for formatting and integration, including:

**Custom Station Data**: User-prepared station data formatted according to standard specifications and converted to NetCDF format.

### Category C Custom Datasets

Based on these two categories of data, we also organize a category of **custom hydrological datasets**, which are datasets constructed for specific research needs based on agreed standard formats.

## Features and Highlights

### Unified Data Management

**hydrodatasource** provides standardized methods for:

- Structuring datasets according to predefined conventions.
- Integrating various data sources into a unified framework.
- Supporting data access and processing for hydrological modeling.

### Compatibility with Local and Cloud Resources

- **Public Data**: Supports data format conversion and local file operations.
- **Non-Public Data**: Provides tools to format and integrate user-prepared data.

### Modular Design

The repository structure supports diverse workflows, including:

1. **Category A Datasets**: Tools to organize and access public hydrological datasets.
2. **Category B Data**: Custom tools to clean and process station, reservoir, and basin time-series data.
3. **Category C Custom Datasets**: Support for reading data in defined standard dataset formats.

### Other Interactions

**hydrodatasource** interacts with the following components:

- [**hydrodataset**](https://github.com/OuyangWenyu/hydrodataset): Provides necessary support for accessing public watershed hydrological modeling datasets for hydrodatasource.
- [**HydroDataCompiler**](https://github.com/iHeadWater/HydroDataCompiler): Supports semi-automated processing of non-public and custom data (currently not public).

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
├── ClassA
  ├── 1st_origin
  ├── 2nd_process
├── ClassB
  ├── 1st_origin
  ├── 2nd_process
├── ClassC
```

- **`1st_origin`**: Raw data, often from proprietary sources, in unified formats.
- **`2nd_process`**: Intermediate results after initial processing and data ready for analysis or modeling.

### Data Reading

The data reading code is mainly located in the reader folder. Currently, the main interface functions provided are:

- Reading GRDC, GAGES, CRD and other datasets
- Reading custom station data
- Reading custom datasets for hydrological modeling

We will provide more detailed documentation in the future.