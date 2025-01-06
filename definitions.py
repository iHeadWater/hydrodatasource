"""
Author: Wenyu Ouyang
Date: 2024-12-29 15:12:55
LastEditTime: 2025-01-03 09:16:41
LastEditors: Wenyu Ouyang
Description: We set some project-wide definitions here so that we can easily unify the paths in the project
FilePath: \hydradatasource\definitions.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

# NOTE: create a file in root directory -- definitions_private.py,
# then copy the code after 'except ImportError:' to definitions_private.py
# and modify the paths as your own paths in definitions_private.py
import os

from hydrodataset import SETTING

try:
    import definitions_private

    PROJECT_DIR = definitions_private.PROJECT_DIR
    RESULT_DIR = definitions_private.RESULT_DIR
    DATASET_DIR = definitions_private.DATASET_DIR
except ImportError:
    # point to this project
    PROJECT_DIR = os.getcwd()
    # where to put results
    RESULT_DIR = r"C:\Users\wenyu\code\hydrodatasource\results"
    # where are the data sources
    DATASET_DIR = SETTING["local_data_path"]["basins-origin"]
