"""
Author: Wenyu Ouyang
Date: 2023-10-31 20:56:23
LastEditTime: 2024-03-28 08:39:15
LastEditors: Wenyu Ouyang
Description: Test funcs for reading stations' data
FilePath: \hydrodatasource\tests\test_reader_stations.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import os
import pandas as pd

from hydrodatasource.reader.stations import huanren_preprocess


def test_huanren_preprocess(tmp_path):
    # create a temporary directory for testing
    test_dir = tmp_path / "test_data"
    test_dir.mkdir()

    # create a test excel file
    test_file = test_dir / "test_file.xlsx"
    test_data = pd.DataFrame(
        {"date": ["2022-01-01", "2022-01-02", "2022-01-03"], "flow": [1.0, 2.0, 3.0]}
    )
    test_data.to_excel(test_file, index=False)

    # set up local data path and run function
    os.environ["LOCAL_DATA_PATH"] = str(test_dir)
    huanren_preprocess()

    # check that the output file was created and has the correct data
    output_file = test_dir / "huanren" / "tidy.csv"
    assert output_file.exists()
    output_data = pd.read_csv(output_file)
    assert output_data.equals(test_data)
