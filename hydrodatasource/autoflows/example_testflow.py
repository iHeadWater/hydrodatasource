"""
Author: Yang Wang
Date: 2024-02-26 08:20:12
LastEditTime: 2024-03-28 08:38:38
LastEditors: Wenyu Ouyang
Description: A test case for the auto-flow
FilePath: \hydrodata\hydrodatasource\autoflows\example_testflow.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""
import pytest

from hydrodatasource.downloader.gfs_downloader_auto import gfs_downloader_00
from hydrodatasource.downloader.gpm_downloader_auto import gpm_downloader
from hydrodatasource.processor.gpm_gfs import make_merge_dataset


# pytest -n=2, pip install pytest-xdist
@pytest.mark.xdist_class
class TestGpmGfsMerge:
    # 我们在Case上采用@pytest.mark. + 分组名称，就相当于该方法被划分为该分组中
    # 注意：一个分组可以有多个方法，一个方法也可以被划分到多个分组中
    @pytest.mark.download
    def test_download_gpm(self):
        gpm_downloader(
            "2022-01-01",
            "2022-01-02",
        )

    @pytest.mark.download
    def test_download_gfs(self):
        gfs_downloader_00(
            "2022-01-01",
            "2022-01-02",
        )

    @pytest.mark.merge
    # 按理说，download gpm和gfs才应该划归一个类，由xdist并行执行
    def test_merge_gpm_gfs(self):
        make_merge_dataset(
            "2022-01-01",
            "2022-01-02",
        )
