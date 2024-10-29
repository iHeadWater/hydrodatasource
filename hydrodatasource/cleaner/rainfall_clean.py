import pandas as pd


def data_check_yearly(
    df_era5land, df_station, year_range=[2010, 2024], diff_range=[0.8, 1.2]
):
    """Calculate the difference of precipitation between era5land and station
    and return the station_lst whose pcrp value is in the diff_range

    Parameters
    ----------
    df_era5land : _type_
        _description_
    df_station : _type_
        _description_
    year_range : list, optional
        _description_, by default [2010, 2024]
    diff_range : list, optional
        the range of difference between the station and the era5land data, by default [0.8, 1.2]

    Returns
    -------
    station_lst: list
        the chosen stations
    """
    station_lst = []
    # save the lst satisfying the criterion into a csv file
    staton_lst_df = pd.DataFrame(columns=["station_id"])
    return staton_lst_df


def data_check_daily_extreme(station_lst, climate_extreme_value, station_extremes):
    """
    Check if the values are in the reasonable range
    values larger than extreme values would be treated as anomaly values

    Parameters
    ----------
    station_lst : _type_
        chosen stations from the data_check_yearly
    climate_extreme_value
        the climate extreme value in a region, which should be calculated according to era5land data
    station_extremes

    Returns
    -------
    df_anomaly_stations_periods: df

    """
    df_anomaly_stations_periods = pd.DataFrame(columns=["STCD", "TM", "DRP"])
    return df_anomaly_stations_periods


def data_check_time_series(station_lst, rule_dict):
    """
    Check if the time series of the station is reasonable

    Parameters
    ----------
    station_lst : _type_
        chosen stations from the data_check_yearly
    rule_dict: dict
        the rule dict, which contains the rules for the data

    Returns
    -------
    df_anomaly_stations_periods: pd.DataFrame

    """
    for key, value in rule_dict.items():
        if key == "data_gradient":
            gradient_value = value
        elif key == "continuous_little_rain":
            little_rain_value = value
        else:
            raise ValueError("This rule is not supported yet!!!")
    # each column:
    df_anomaly_stations_periods = pd.DataFrame(
        columns=["STCD", "TM", "DRP", "ANOM_REASON"]
    )
    return df_anomaly_stations_periods


def time_series_model_train(station_lst, config):
    """
    Train the time series model and save the model

    Parameters
    ----------
    station_lst : _type_
        chosen stations after the former three steps
    config: dict
        the configuration of the model

    Returns
    -------
    model: _type_
        the trained model

    """
    # remember to save the model as pth file
    return model


def time_series_model_evaluate(station_lst, diff_range=150):
    """
    Predict the time series of the station and calculate the difference between the predicted value and the real value

    Parameters
    ----------
    station_lst : _type_
        chosen stations after the former three steps
    diff_range: _type_


    Returns
    -------
    df_anomaly_stations_periods: pd.DataFrame


    """
    return df_ts_series_anomaly_ranges


def data_check_ts_range_anomaly(station_lst, df_ts_series_anomaly_ranges):
    """ """
    return df_anomaly_stations_periods
