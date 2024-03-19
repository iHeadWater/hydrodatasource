import pytest
from hydrodata.cleaner.rain_flow_division import Dcma_esr


@pytest.fixture
def dcma_esr():
    dcma_esr = Dcma_esr(
        rain_min=0.02,
        max_window=100,
        flow_max=100,
        duration_max=2400,  # /hour
        multiple=1,
        plot=True,
        rain_file='/home/liutianxv/测试数据/filtered_rain_average.csv',  # df['TM','rain']
        flow_file='/home/liutianxv/测试数据/biliu_inq_interpolated.csv',  # df['TM','INQ']
        save_path='/home/liutianxv/测试数据/biliu_flow_division.csv'
    )
    return dcma_esr


def test_rain_flow_division(dcma_esr):
    dcma_esr.process_rain_flow_division()
