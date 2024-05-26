import os
import pandas as pd
from hydrodatasource.downloader.iowa_stations_download import gen_iowa_link, download_from_link


def test_dload_from_stations_csv():
    sta_ids = pd.read_csv('your_prefer_local_path.csv')
    for i in range(len(sta_ids)):
        network = sta_ids['NETWORK'][i]
        id = sta_ids['ID'][i]
        csv_path = f'iowa_datasets/{network}_{id}.csv'
        if not os.path.exists(csv_path):
            request_link = gen_iowa_link(network, id, '2015-01-01', '2024-05-26')
            if request_link is not None:
                pd_df = download_from_link(request_link)
            else:
                continue
            if len(pd_df) != 0:
                pd_df.to_csv(csv_path)
