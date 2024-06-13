import glob
import os.path
import pathlib

import rasterio
from rasterio.merge import merge
import whitebox


def test_concat_multiple_pics():
    file_paths = glob.glob('/ftproot/dems/*.hgt', recursive=True)
    wbt = whitebox.WhiteboxTools()
    output_tif = os.path.join(os.path.abspath(os.curdir), 'merged_hjlm.tif')
    wbt.mosaic(output=output_tif, inputs=file_paths)


def d8_pointer(dem, des):
    """
    划分子流域

    Todo:
        - 填缺: [fill_missing_data](https://whiteboxgeo.com/manual/wbt_book/available_tools/geomorphometric_analysis.html?#fillmissingdata)
        - 填洼: [fill_depressions](https://whiteboxgeo.com/manual/wbt_book/available_tools/hydrological_analysis.html#FillDepressions)
        - 流向: [d8_pointer](https://whiteboxgeo.com/manual/wbt_book/available_tools/hydrological_analysis.html#D8Pointer)

    Args:
        dem (str): 必选，DEM文件名，格式为.tif
        des (str): 必选，目标文件夹
    """

    # dem = os.path.abspath(dem)
    # des = os.path.abspath(des)
    wbt = whitebox.WhiteboxTools()
    work_dir = pathlib.Path(dem).parent.absolute()
    wbt.set_working_dir(work_dir)
    wbt.fill_missing_data(i=dem, output=os.path.join(work_dir, "fillmissingdata.tif"), filter=11, weight=2.0, no_edges=True)
    wbt.fill_depressions_wang_and_liu("fillmissingdata.tif", "filldepression.tif", fix_flats=True, flat_increment=None)
    wbt.d8_pointer("filldepression.tif","d8pointer.tif", esri_pntr=False)


def test_d8_pointer():
    d8_pointer(dem='DEM.tif', des='.')
