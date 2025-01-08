"""
Author: liutiaxqabs 1498093445@qq.com
Date: 2024-04-19 14:00:16
LastEditors: Wenyu Ouyang
LastEditTime: 2025-01-08 10:00:18
FilePath: \hydrodatasource\hydrodatasource\cleaner\rsvr_inflow_cleaner.py
Description: calculate streamflow from reservoir timeseries data
"""

import logging
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from hydrodatasource.configs.table_name import RSVR_TS_TABLE_COLS


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class ReservoirInflowBacktrack:
    def __init__(self, data_folder, output_folder):
        """
        Back-calculating inflow of reservior

        Parameters
        ----------
        data_folder : str
            the folder of reservoir data
        output_folder : _type_
            where we put inflow data
        """
        self.data_folder = data_folder
        self.output_folder = output_folder
        logging.info(
            "Please make sure the data is in the right format. We are checking now ..."
        )
        self._check_file_format(data_folder)

    def _check_file_format(self, folder_path):
        """
        Check if the files in the given folder match the specified format.

        Parameters
        ----------
        folder_path : str
            The path of the folder to check.

        Raises
        ----------
        ValueError
            If a file name does not match the specified format, an error is raised with the specific file name.
        """
        pattern = re.compile(r".+_rsvr_data\.csv$")
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if not pattern.match(file):
                    raise ValueError(f"File name does not match the format: {file}")
                file_path = os.path.join(root, file)
                try:
                    df = pd.read_csv(file_path)
                except Exception as e:
                    raise ValueError(f"Unable to read file: {file}, error: {e}") from e

                if any(column not in df.columns for column in RSVR_TS_TABLE_COLS):
                    raise ValueError(
                        f"File content does not match the format: {file}, missing columns: {set(RSVR_TS_TABLE_COLS) - set(df.columns)}"
                    )
        logging.info("All files are in the right format.")

    def _rsvr_rolling_window_abnormal_rm(
        self, df, var_col="RZ", threshold=50, window_size=5
    ):
        """
        Detect and remove abnormal reservoir water level data using a rolling window.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame containing the reservoir data.
        var_col : str
            the column to check, by default "RZ"
        threshold: float
            the threshold to remove the abnormal data
        window_size : int
            The size of the rolling window.

        Returns
        -------
        pd.DataFrame
            The DataFrame with an additional column indicating abnormal data.
        """

        # Calculate the median of the rolling window
        df["median"] = df[var_col].rolling(window=window_size, center=True).median()
        # Calculate the difference between the current value and the median
        df["diff_median"] = abs(df[var_col] - df["median"])
        # Mark as abnormal if the difference exceeds the threshold
        df["set_nan"] = df["diff_median"] > threshold
        # Set abnormal values to NaN
        df.loc[df["set_nan"], var_col] = np.nan
        return df

    def _rsvr_conservative_abnormal_rm(self, df, var_col="RZ", threshold=10):
        """
        Remove abnormal data from the specified column using robust Z-Score method

        Parameters
        ----------
        df : pd.DataFrame
            The data.
        var_col : str, optional
            The column to check, by default "RZ".
        threshold : float, optional
            The threshold to remove the abnormal data, by default 10.
        """
        median = np.median(df[var_col])
        mad = np.median(np.abs(df[var_col] - median))
        if mad == 0:
            z_scores = np.zeros_like(df[var_col])
        else:
            z_scores = (df[var_col] - median) / mad
        df["set_nan"] = np.abs(z_scores) > threshold
        df.loc[df["set_nan"], var_col] = np.nan
        return df


   def _save_fitted_zw_curve(self, df, quadratic_fit_curve_coeff, output_folder):
        """Save a plot of the RZ and W points along with the fitted curve so that
        the relationship between RZ and W can be visualized and verified

        Parameters
        ----------
        df : pd.DataFrame
            _description_
        quadratic_fit_curve_coeff : _type_
            a list of coefficients of the quadratic fit curve
        output_folder : _type_
            _description_
        """
        plt.figure(figsize=(14, 7))
        plt.scatter(df["RZ"], df["W"], label="Data Points")
        rz_range = np.linspace(df["RZ"].min(), df["RZ"].max(), 100)
        w_fit = (
            quadratic_fit_curve_coeff[0] * rz_range**2
            + quadratic_fit_curve_coeff[1] * rz_range
            + quadratic_fit_curve_coeff[2]
        )
        plt.plot(rz_range, w_fit, color="red", label="Fitted Curve")
        plt.xlabel("RZ (m)")
        plt.ylabel("W (10^6 m^3)")
        plt.legend()
        plt.title("RZ vs W with Fitted Curve")
        plot_path = os.path.join(output_folder, "fit_zw_curve.png")
        plt.savefig(plot_path)

    def _plot_var_before_after_clean(
        self,
        df_origin,
        df,
        plot_column,
        plot_path,
        label_orginal="Original Reservoir Storage",
        label_cleaned="Cleaned Reservoir Storage",
        ylab="Reservoir Storage (10^6 m^3)",
        title="Reservoir Storage Analysis with Outliers Removed",
    ):
        """Plot the original and cleaned Reservoir Storage data for comparison

        Parameters
        ----------
        df_origin : str
            the original data
        df : pd.DataFrame
            the cleaned data
        plot_path : str
            where to save the plot
        plot_column: str
            the column to show; note same name for df and df_origin
        """
        plt.figure(figsize=(14, 7))

        # 绘制原始数据
        plt.plot(
            pd.to_datetime(df_origin["TM"]),
            df_origin[plot_column],
            label=label_orginal,
            color="blue",
            linestyle="--",
        )

        # 绘制清洗后的数据
        plt.plot(
            pd.to_datetime(df["TM"]),
            df[plot_column],
            label=label_cleaned,
            color="red",
        )

        plt.xlabel("Time")
        plt.ylabel(ylab)
        plt.title(title)
        plt.legend()

        # 保存图像到与CSV文件相同的目录
        plt.savefig(plot_path)

    def clean_w(self, file_path, output_folder):
        """
        Remove abnormal reservoir capacity data

        Parameters
        ----------
        file_path : str
            Path to the input file
        output_folder : str
            Path to the output folder

        Returns
        -------
        str
            Path to the cleaned data file
        """
        data = pd.read_csv(file_path)

        # remove abnormal reservoir water level data
        # 50 means 50m difference between the current value and the median
        data = self._rsvr_conservative_abnormal_rm(data, "RZ", threshold=10)
        # 100 means 0.1 billion m^3 difference between the current value and the median
        data = self._rsvr_conservative_abnormal_rm(data, "W", threshold=100)

        # 输出被设置为 NaN 的行
        logging.debug(data[data["set_nan"]])

        # 保存被设置为 NaN 的行到 CSV 文件
        data[data["set_nan"]].to_csv(
            os.path.join(output_folder, "库容异常的数据行.csv"), index=False
        )
        try:
            # 拟合库容曲线
            # 只提取 RZ 和 W 列中同时非 NaN 的行
            valid_data = data.dropna(subset=["RZ", "W"])

            # 执行二次拟合，计算 RZ 和 W 之间的关系
            coefficients = np.polyfit(valid_data["RZ"], valid_data["W"], 2)
            # Plot RZ and W points along with the fitted curve
            self._save_fitted_zw_curve(valid_data, coefficients, output_folder)
            # 根据拟合的多项式关系更新 W 列
            data["W"] = (
                coefficients[0] * data["RZ"] ** 2
                + coefficients[1] * data["RZ"]
                + coefficients[2]
            )
        except np.linalg.LinAlgError:
            print("SVD did not converge during polynomial fitting, skipping this step.")

        cleaned_path = os.path.join(output_folder, "去除库容异常的数据.csv")
        data.to_csv(cleaned_path)
        original_data = pd.read_csv(file_path)
        plot_path = os.path.join(output_folder, "rsvr_w_clean.png")
        self._plot_var_before_after_clean(original_data, data, "W", plot_path)
        return cleaned_path

    def back_calculation(self, clean_w_path, original_file, output_folder):
        """Back-calculate inflow from reservoir storage data
        NOTE: each time has three columns: I Q W -- I is the inflow, Q is the outflow, W is the reservoir storage
        Generally, in sql database, a time means the end of previous time period
        For example, a hourly database, 13:00 means 12:00-13:00 period because the data is GOT at 13:00 (we cannot observe future)
        Hence, for this function, W means the storage at the end of the time period, I and Q means the inflow and outflow of the time period
        So we need to use W of the previous time as the initial water storage of the time period.
        Hence, I1 = Q1 + (W1 - W0)


        Parameters
        ----------
        data_path : str
            the path to the cleaned_w_data file
        original_file: str
            the path to the original file
        output_folder : str
            where to save the back calculated data

        Returns
        -------
        str
            the path to the result file
        """
        data = pd.read_csv(clean_w_path)
        # 将时间列转换为日期时间格式
        data["TM"] = pd.to_datetime(data["TM"])
        # diff means the difference between this time and the previous time -- the first will be 0 as fillna(0)
        data["Time_Diff"] = data["TM"].diff().dt.total_seconds().fillna(0)
        data["INQ_ACC"] = data["OTQ"] + (10**6 * (data["W"].diff() / data["Time_Diff"]))
        data["INQ"] = data["INQ_ACC"]
        # data["Month"] = data["TM"].dt.month
        logging.debug(data)
        back_calc_path = os.path.join(
            output_folder, f"{int(data['STCD'][0])}_径流直接反推数据.csv"
        )
        data[RSVR_TS_TABLE_COLS].to_csv(back_calc_path)
        # plot the inflow data and compare with the original data
        original_data = pd.read_csv(original_file)
        self._plot_var_before_after_clean(
            original_data,
            data,
            "INQ",
            os.path.join(output_folder, "inflow_comparison.png"),
            label_orginal="Original Inflow",
            label_cleaned="Back-calculated Inflow",
            ylab="Inflow (m^3/s)",
            title="Inflow Analysis with Back-calculation",
        )
        return back_calc_path

    def delete_negative_inq(
        self,
        inflow_data_path,
        original_file,
        output_folder,
        negative_deal_window=7,
        negative_deal_stride=4,
    ):
        """remove negative inflow values with a rolling window
        the negative value will be adjusted to positvie ones to make the total inflow consistent
        for example,  1, -1, 1, -1 will be adjusted to 0, 0, 0, 0 so that wate balance is kept
        but note that as the window has stride, maybe the final few values will not be adjusted

        Parameters
        ----------
        inflow_data_path : str
            the data file after back_calculation
        original_file : str
            the original file
        output_folder : str
            where to save the data
        negative_deal_window : int, optional
            the window to deal with negative values, by default 7
        negative_deal_stride : int, optional
            the stride of window, by default 4

        Returns
        -------
        str
            the path to the result file
        """
        # 读取CSV文件到DataFrame
        df = pd.read_csv(inflow_data_path)
        # 将'TM'列转换为日期时间格式并设置为索引
        df["TM"] = pd.to_datetime(df["TM"])

        # 设置调整后的时间为索引
        df = df.set_index("TM")

        logging.debug(df["INQ"].sum())
        # Ensure the 'INQ' column is numeric. If a value cannot be parsed as a number, the errors="coerce" parameter will set it to NaN (i.e., missing value)
        df["INQ"] = pd.to_numeric(df["INQ"], errors="coerce")

        def adjust_window(window):
            """adjust window for delete negative inflow values

            Parameters
            ----------
            window : pd.Series
                the data in the window

            Returns
            -------
            _type_
                _description_
            """
            if window.count() == 0:
                return window  # 如果窗口内全是NaN，返回原窗口

            # 移除负值
            positive_values = window[window > 0]
            negative_values = window[window < 0]

            # 计算正负值的总和
            pos_sum = positive_values.sum()
            neg_sum = abs(negative_values.sum())  # 负值的绝对值和

            # 计算需要调整的比例
            if pos_sum > 0:
                adjust_factor = neg_sum / pos_sum
                # 调整正值
                adjusted_values = positive_values - (positive_values * adjust_factor)
            else:
                adjusted_values = positive_values  # 如果没有正值可用于调整，保持原样

            # 更新窗口的值
            window[window > 0] = adjusted_values
            window[window <= 0] = 0

            return window

        def rolling_with_stride(df, column, window_size, stride, func):
            # 遍历数据，步长为stride
            for i in range(0, len(df) - window_size + 1, stride):
                window_indices = range(i, i + window_size)
                df.loc[df.index[window_indices], column] = func(
                    df.loc[df.index[window_indices], column]
                )

        # 应用滚动窗口函数，这里设置步幅为4，窗口大小为7
        rolling_with_stride(
            df,
            "INQ",
            window_size=negative_deal_window,
            stride=negative_deal_stride,
            func=adjust_window,
        )
        path = os.path.join(
            output_folder, f"{int(df['STCD'][0])}_水量平衡后的日尺度反推数据.csv"
        )

        df["TM"] = df.index.strftime("%Y-%m-%d %H:%M:%S")
        df[RSVR_TS_TABLE_COLS].to_csv(path, index=False)
        # plot the inflow data and compare with the original data
        original_data = pd.read_csv(original_file)
        self._plot_var_before_after_clean(
            original_data,
            df,
            "INQ",
            os.path.join(output_folder, "inflow_comparison_after_negative.png"),
            label_orginal="Original Inflow",
            label_cleaned="Inflow After Negative Removal",
            ylab="Inflow (m^3/s)",
            title="Inflow Analysis with Negative Removal",
        )
        return path

    def insert_inq(self, inflow_data_path, original_file, output_folder):
        """make inflow data as hourly data as original data is not strictly hourly data
        and insert inq with linear interpolation

        Parameters
        ----------
        inflow_data_path : str
            the data file after delete negative inflow values
        original_file : str
            the original file
        output_folder : str
            where to save the data

        Returns
        -------
        str
            the path to the result file
        """
        # 读取CSV文件到DataFrame
        df = pd.read_csv(inflow_data_path)
        # 将'TM'列转换为日期时间格式并设置为索引
        df["TM"] = pd.to_datetime(df["TM"])
        # 设置调整后的时间为索引
        df = df.set_index("TM")
        # 确保'INQ'列是数值类型
        df["INQ"] = pd.to_numeric(df["INQ"], errors="coerce")

        # 生成从开始日期到结束日期的完整时间序列，按小时
        date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq="h")
        complete_df = pd.DataFrame(index=date_range)

        # Perform a full outer join of the original data with the complete time series table, so that some sub-hourly data could still be saved here
        df = complete_df.join(df, how="outer")
        # before we interpolate, we need to guarantee all data has same time interval
        df = df.resample("H").asfreq()

        # Ensure INQ values are not less than 0 -- mainly for final few values as previous steps may not adjust them
        df["INQ"] = df["INQ"].where(df["INQ"] >= 0, np.nan)

        # 使用线性插值
        # 插值前检查连续缺失是否超过7天（7*24小时）
        df = linear_interpolate_wthresh(df)

        result_path = os.path.join(output_folder, f"{int(df['STCD'][0])}_rsvr_data.csv")

        logging.debug("水量平衡的小时尺度滑动平均反推数据：输出行名称")
        logging.debug(df.columns)
        df["TM"] = df.index.strftime("%Y-%m-%d %H:%M:%S")
        df["STCD"] = df["STCD"].dropna().iloc[0]
        # 最后一步转换为整数再转换为字符串
        df["STCD"] = df["STCD"].astype(int).astype(str)
        logging.debug(df["STCD"])
        df[RSVR_TS_TABLE_COLS].to_csv(result_path, index=False)
        # plot the inflow data and compare with the original data
        original_data = pd.read_csv(original_file)
        self._plot_var_before_after_clean(
            original_data,
            df,
            "INQ",
            os.path.join(output_folder, "inflow_comparison_after_interpolation.png"),
            label_orginal="Original Inflow",
            label_cleaned="Inflow After Interpolation",
            ylab="Inflow (m^3/s)",
            title="Inflow Analysis with Interpolation",
        )
        return result_path

    def process_backtrack(self):
        for file in tqdm(os.listdir(self.data_folder)):
            file_path = os.path.join(self.data_folder, file)
            output_folder = os.path.join(self.output_folder, file[:-4])
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            # Process each file step by step
            # 去除库容异常
            cleaned_data_file = self.clean_w(file_path, output_folder)
            # 公式计算反推
            back_data_file = self.back_calculation(
                cleaned_data_file, file_path, output_folder
            )
            # 去除反推异常值
            nonegative_data_file = self.delete_negative_inq(
                back_data_file, file_path, output_folder
            )
            # 插值平衡
            self.insert_inq(nonegative_data_file, file_path, output_folder)


def linear_interpolate_wthresh(df, column="INQ", threshold=168):
    """linear interpolation for inflow data with a threshod

    Parameters
    ----------
    df : pd.DataFrame
        pandas DataFrame containing the inflow data
    column : str, optional
        the chosen column, by default "INQ"
    threshold : int, optional
        under this threshold we interpolate, by default 168,
        if the missing data is larger than 7 days, we didn't interpolate it

    Returns
    -------
    pd.DataFrame
        DataFrame with interpolated values
    """
    # Calculate the gap lengths of missing values
    mask = df[column].isna()
    gap_lengths = []
    gap_length = 0
    for is_na in mask:
        if is_na:
            gap_length += 1
        else:
            if gap_length > 0:
                gap_lengths.extend([gap_length] * gap_length)
                gap_length = 0
            gap_lengths.append(0)
    if gap_length > 0:
        gap_lengths.extend([gap_length] * gap_length)

    # Convert gap lengths to Series
    gap_lengths = pd.Series(gap_lengths, index=df.index)
    # Only interpolate missing values with gaps less than the threshold, and set limit_direction to 'both' to ensure extrapolation
    df.loc[mask & (gap_lengths <= threshold), column] = df[column].interpolate(
        limit_direction="both"
    )

    return df
