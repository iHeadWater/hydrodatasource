import os

import pytest

from hydroprivatedata import config
from hydroprivatedata.minio_api import minio_sync_files, boto3_sync_files

pytest_plugins = ('pytest_asyncio',)


@pytest.mark.asyncio
async def test_sync_data():
    s3_client = config.s3
    mc_client = config.mc
    await minio_sync_files(mc_client, 'forestbat-private',
                           local_path=os.path.join(config.LOCAL_DATA_PATH, 'forestbat_test'))
    await boto3_sync_files(s3_client, 'forestbat-private',
                           local_path=os.path.join(config.LOCAL_DATA_PATH, 'forestbat_test_1'))
