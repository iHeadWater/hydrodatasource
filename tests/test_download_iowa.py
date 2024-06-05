"""
Author: liutiaxqabs 1498093445@qq.com
Date: 2024-05-28 10:24:37
LastEditors: liutiaxqabs 1498093445@qq.com
LastEditTime: 2024-06-03 16:35:13
FilePath: /hydrodatasource/tests/test_download_iowa.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
"""

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from hydrodatasource.downloader.iowa_stations_download import (
    gen_iowa_link,
    download_from_link,
)

def test_dload_from_stations_csv():
    for slice in np.arange(8, 0, -2):
        file_path = f"/ftproot/iowa_stations_table/iowa_stations_{slice}.csv"
        print(file_path)
        try:
            sta_ids = pd.read_csv(file_path, encoding="utf-8")
            for i in tqdm(range(len(sta_ids)), desc=f'Processing slice {slice}'):
                network = sta_ids["NETWORK"][i]
                id = sta_ids["ID"][i]
                csv_path = f"/ftproot/iowa_stations/{network}_{id}.csv"
                print('当前正在执行保存：' + csv_path)
                if not os.path.exists(csv_path):
                    request_link = gen_iowa_link(network, id, "2015-01-01", "2024-05-26")
                    if request_link is not None:
                        pd_df = download_from_link(request_link)
                    else:
                        continue
                    if len(pd_df) != 0:
                        pd_df.to_csv(csv_path)
        except FileNotFoundError:
            print(f"File not found: {file_path}")
