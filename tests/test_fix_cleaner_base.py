"""Regression tests for the Cleaner base-class bug.

Bug (review finding): ``Cleaner.__init__`` (hydrodatasource/cleaner/cleaner.py)
never initializes ``origin_df`` / ``processed_df``, and ``read_data()`` is a
no-op (``pass``) even though its docstring claims it "reads data and stores it
in origin_df". As a result:

  * ``StreamflowCleaner.anomaly_process`` (streamflow_cleaner.py:393-407)
    reads ``self.origin_df`` and writes ``self.processed_df``
  * ``WaterlevelCleaner.anomaly_process`` (waterlevel_cleaner.py:79-94)
    does the same

...so every call raises ``AttributeError`` and both cleaner classes are
effectively dead. The documented usage is
``scripts/chinese_rsvr_inflow_preprocessing.py:41-53`` — construct a
``StreamflowCleaner`` with a CSV path, then call ``anomaly_process(["EMA"])``.

Contract pinned here (implementation-agnostic, no real data required):
  * After construction a cleaner exposes ``origin_df`` and ``processed_df``
    as pandas DataFrames.
  * Given a minimal CSV with the columns the methods expect, ``anomaly_process``
    completes without raising ``AttributeError`` and stores the cleaned series
    back onto ``processed_df``.

RED on the current (unfixed) code: construction exposes no such attributes and
``anomaly_process`` raises ``AttributeError`` immediately.
"""


def _write_streamflow_csv(tmp_path):
    """Write a minimal streamflow CSV (TM + INQ columns) and return its path."""
    import pandas as pd

    csv_path = tmp_path / "streamflow_input.csv"
    df = pd.DataFrame(
        {
            "TM": pd.date_range("2023-01-01", periods=48, freq="D"),
            "INQ": [100, 110, 120, 130, 95, 105, 115, 125] * 6,
        }
    )
    df.to_csv(csv_path, index=False)
    return csv_path


def _write_waterlevel_csv(tmp_path):
    """Write a minimal waterlevel CSV (TM + Z columns) and return its path."""
    import pandas as pd

    csv_path = tmp_path / "waterlevel_input.csv"
    df = pd.DataFrame(
        {
            "TM": pd.date_range("2023-01-01", periods=48, freq="D"),
            "Z": [100.0, 100.5, 101.0, 100.8, 101.2, 101.5, 101.3, 101.8] * 6,
        }
    )
    df.to_csv(csv_path, index=False)
    return csv_path


def test_cleaner_base_exposes_origin_and_processed_df_after_construction(tmp_path):
    """The base Cleaner exposes origin_df / processed_df after construction."""
    import pandas as pd

    from hydrodatasource.cleaner.cleaner import Cleaner

    csv_path = _write_streamflow_csv(tmp_path)
    cleaner = Cleaner(data_folder=str(csv_path))

    assert isinstance(cleaner.origin_df, pd.DataFrame)
    assert isinstance(cleaner.processed_df, pd.DataFrame)


def test_streamflow_cleaner_exposes_origin_and_processed_df_after_construction(
    tmp_path,
):
    """StreamflowCleaner exposes origin_df / processed_df after construction."""
    import pandas as pd

    from hydrodatasource.cleaner.streamflow_cleaner import StreamflowCleaner

    csv_path = _write_streamflow_csv(tmp_path)
    cleaner = StreamflowCleaner(data_folder=str(csv_path))

    assert isinstance(cleaner.origin_df, pd.DataFrame)
    assert isinstance(cleaner.processed_df, pd.DataFrame)


def test_streamflow_cleaner_anomaly_process_moving_average_writes_processed_df(
    tmp_path,
):
    """anomaly_process(['moving_average']) runs without AttributeError."""
    from hydrodatasource.cleaner.streamflow_cleaner import StreamflowCleaner

    csv_path = _write_streamflow_csv(tmp_path)
    cleaner = StreamflowCleaner(data_folder=str(csv_path))

    cleaner.anomaly_process(["moving_average"])

    assert "moving_average" in cleaner.processed_df.columns
    assert len(cleaner.processed_df) == 48


def test_streamflow_cleaner_anomaly_process_ema_writes_processed_df(tmp_path):
    """anomaly_process(['EMA']) — the documented usage — runs without AttributeError."""
    from hydrodatasource.cleaner.streamflow_cleaner import StreamflowCleaner

    csv_path = _write_streamflow_csv(tmp_path)
    cleaner = StreamflowCleaner(data_folder=str(csv_path))

    cleaner.anomaly_process(["EMA"])

    assert "EMA" in cleaner.processed_df.columns
    assert len(cleaner.processed_df) == 48


def test_waterlevel_cleaner_exposes_origin_and_processed_df_after_construction(
    tmp_path,
):
    """WaterlevelCleaner exposes origin_df / processed_df after construction."""
    import pandas as pd

    from hydrodatasource.cleaner.waterlevel_cleaner import WaterlevelCleaner

    csv_path = _write_waterlevel_csv(tmp_path)
    cleaner = WaterlevelCleaner(data_folder=str(csv_path))

    assert isinstance(cleaner.origin_df, pd.DataFrame)
    assert isinstance(cleaner.processed_df, pd.DataFrame)


def test_waterlevel_cleaner_anomaly_process_roll_writes_processed_df(tmp_path):
    """anomaly_process(['roll']) runs without AttributeError and writes results."""
    from hydrodatasource.cleaner.waterlevel_cleaner import WaterlevelCleaner

    csv_path = _write_waterlevel_csv(tmp_path)
    cleaner = WaterlevelCleaner(data_folder=str(csv_path))

    cleaner.anomaly_process(["roll"])

    assert len(cleaner.processed_df.columns) >= 1
    assert len(cleaner.processed_df) == 48
