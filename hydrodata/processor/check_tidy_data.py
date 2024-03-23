"""
Author: Wenyu Ouyang
Date: 2023-10-25 15:16:21
LastEditTime: 2024-03-23 19:14:52
LastEditors: Wenyu Ouyang
Description: To check if user's data format is correct
FilePath: \hydrodata\hydrodata\processor\check_tidy_data.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""
# 定义测站类型代码表
STATION_TYPE_CODES = {
    "气象站": "MM",
    "雨量站": "PP",
    "蒸发站": "BB",
    "河道水文站": "ZQ",
    "堰闸水文站": "DD",
    "河道水位站": "ZZ",
    "潮位站": "TT",
    "水库水文站": "RR",
    "泵站": "DP",
    "地下水站": "ZG",
    "墒情站": "SS",
    "分洪水位站": "ZB",
}


def format1_to_tidy(df):
    """
    Format 1 -- from huanren:
    - The data is structured with a 'date' column and multiple variable columns.
    - Each row represents data for a specific date.
    - The variable columns contain values for different measurements on that date.

    Example:
        date     | z_0   | v_0  | ...
    ------------|-------|------|-----
    1967-01-01  | 244.57| 0.0  | ...
    1967-01-02  | 244.58| 0.0  | ...
    ...         | ...   | ...  | ...

    """
    station_type_code = STATION_TYPE_CODES["水库水文站"]

    # 将宽格式数据转化为长格式
    tidy_df = df.melt(
        id_vars=["date"],
        value_vars=[col for col in df.columns if col != "date"],
        var_name="variable",
        value_name="value",
    )
    tidy_df["station_type"] = station_type_code
    return tidy_df


def format2_to_tidy(df):
    """
    Format 2 -- from huanren:
    - The data is structured with a 'date' column and multiple year columns.
    - Each row represents data for a specific date.
    - The year columns contain values for different measurements on that date for a specific year.

    Example:
        date     | 2014 | 2015 | ...
    ------------|------|------|-----
    1967-01-01  | 46.7 | 11.2 | ...
    1967-01-02  | 33.9 | 11.6 | ...
    ...         | ...  | ...  | ...

    """
    station_type_code = STATION_TYPE_CODES["水库水文站"]

    # 将宽格式数据转化为长格式
    tidy_df = df.melt(
        id_vars=["date"],
        value_vars=[
            col for col in df.columns if col not in ["date", "Unnamed: 6", "Unnamed: 7"]
        ],
        var_name="year",
        value_name="value",
    )
    tidy_df["station_type"] = station_type_code
    return tidy_df


# 定义转换字典
CONVERSION_DICT = {"format1": format1_to_tidy, "format2": format2_to_tidy}


def convert_to_tidy(df, format_type):
    # 使用字典中的函数进行转换
    return CONVERSION_DICT[format_type](df)
