from hydrodata.processor import gpm, gfs, merge


def test_gpm():
    gpm_data = gpm.make_gpm_dataset()
    assert gpm_data is not None
    # return是方便在DEBUG的时候可以直接打印出来数据
    return gpm_data


def test_gfs():
    gfs_data = gfs.make_gfs_dataset()
    assert gfs_data is not None
    return gfs_data


def test_merge():
    merge_data = merge.merge_data()
    assert merge_data is not None
    return merge_data
