from hydrodata.reader import access_fs
from hydrodata.reader.access_fs import gen_refer_and_read_zarr
import hydrodata.configs.config as conf


def test_read_spec():
    # access_fs.spec_path("st_rain_c.csv")
    # gpm_gfs_nc = access_fs.spec_path('grids-interim/86_21401550/gpm_gfs.nc', head='minio')
    print_json = access_fs.spec_path('test/geodata/era5_land/era5_land_.json', head='minio', need_cache=False)


def test_gen_refer():
    obj_path = 'grids-origin/GFS/1/2019/03/03/12/gfs20190303.t12z.nc4.0p25.f001'
    gen_refer_and_read_zarr(obj_path, storage_option=conf.MINIO_PARAM)
