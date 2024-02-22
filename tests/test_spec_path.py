from hydrodata.reader import access_fs

def test_read_spec():
    # access_fs.spec_path(r"C:\Users\Administrator\IdeaProjects\hydrodata\hydrodata\downloader\downloader.py")
    # 只能绝对路径，或者将相对路径拼接为绝对路径
    # access_fs.spec_path('/hydrodata/downloader/downloader.py')
    access_fs.spec_path('grids-interim/86_21401550/gpm_gfs.nc', head='minio')
