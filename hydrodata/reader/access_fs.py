import gzip
import os
import pathlib

import fsspec
import xarray

import hydrodata.configs.config as conf


# 实验性质
def spec_path(url_path: str | os.PathLike, head='local'):
    if head == 'local':
        url_path = pathlib.Path(url_path).resolve()
        with fsspec.open(url_path) as fp:
            fp.read()
    elif head == 'minio':
        url_path = 's3://' + url_path
        with conf.FS.open(url_path) as fp:
            # 这里改成任何读取数据的代码
            zip_fp = gzip.GzipFile(fileobj=fp, mode='rb')
            # 仅作示例，实际上读不通，路径对上就行了
            ds = xarray.open_dataset(zip_fp.read())
    else:
        raise ValueError("head should be 'local' or 'minio'")
