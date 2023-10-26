#!/usr/bin/env python
"""
Author: Wenyu Ouyang
Date: 2023-10-25 15:16:21
LastEditTime: 2023-10-26 08:56:11
LastEditors: Wenyu Ouyang
Description: Tests for preprocess
FilePath: \hydro_privatedata\tests\test_hydroprivatedata.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""
from hydroprivatedata import preprocess


def test_content():
    preprocess.huanren_preprocess()
