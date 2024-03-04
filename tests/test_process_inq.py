import pytest
from hydrodata.cleaner.smooth_inq import Cleaner

@pytest.fixture
def cleaner():
    cleaner = Cleaner(
        file_path="/home/liutianxv/sample_data.csv",
        column_id='ID',
        ID_list=None,
        column_flow='INQ',
        column_time='TM',
        start_time=None,
        end_time=None,
        preprocess=True,
        method='kalman',
        save_path="/home/liutianxv/sample_data.csv",
        plot=True,
        window_size=20,
        cutoff_frequency=0.035,
        iterations=3,
        sampling_rate=1.0,
        time_step=1,
        order=5,
        cwt_row=10
    )
    return cleaner

def test_process_inq(cleaner):
    cleaner.process_inq()
