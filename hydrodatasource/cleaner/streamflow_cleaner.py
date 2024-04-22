'''
Author: liutiaxqabs 1498093445@qq.com
Date: 2024-04-19 14:00:16
LastEditors: liutiaxqabs 1498093445@qq.com
LastEditTime: 2024-04-22 14:56:16
FilePath: /hydrodatasource/hydrodatasource/cleaner/streamflow_cleaner.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from .cleaner import Cleaner
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import cwt, morlet, butter, filtfilt
from scipy.fft import fft, ifft, fftfreq
from scipy.optimize import curve_fit


class StreamflowCleaner(Cleaner):
    def __init__(
        self,
        data_path,
        window_size=5,
        cutoff_frequency=0.01,
        time_step=1.0,
        iterations=3,
        sampling_rate=1.0,
        order=5,
        cwt_row=1,
        *args,
        **kwargs,
    ):
        self.window_size = window_size
        self.cutoff_frequency = cutoff_frequency
        self.time_step = time_step
        self.iterations = iterations
        self.sampling_rate = sampling_rate
        self.order = order
        self.cwt_row = cwt_row
        super().__init__(data_path, *args, **kwargs)

    def data_balanced(self, origin_data, transform_data):
        """
        对一维流量数据进行总量平衡变换。
        :origin_data: 原始一维流量数据。
        :transform_data: 平滑转换后的一维流量数据。
        """
        # Calculate the flow balance factor and keep the total volume consistent
        streamflow_data_before = np.sum(origin_data)
        streamflow_data_after = np.sum(transform_data)
        scaling_factor = streamflow_data_before / streamflow_data_after
        balanced_data = transform_data * scaling_factor

        print(f"Total flow (before smoothing): {streamflow_data_before}")
        print(f"Total flow (after smoothing): {np.sum(balanced_data)}")
        return balanced_data

    def moving_average(self, streamflow_data, window_size=20):
        """
        对流量数据应用滑动平均进行平滑处理，并保持流量总量平衡。
        :window_size: 滑动窗口大小
        :return: 平滑处理后的流量数据
        """
        smoothed_data = np.convolve(
            streamflow_data, np.ones(self.window_size) / self.window_size, mode="same"
        )

        # Apply non-negative constraints
        smoothed_data[smoothed_data < 0] = 0
        return self.data_balanced(streamflow_data, smoothed_data)

    def kalman_filter(self, streamflow_data):
        """
        对流量数据应用卡尔曼滤波进行平滑处理，并保持流量总量平衡。
        :param streamflow_data: 原始流量数据
        """
        A = np.array([[1]])
        H = np.array([[1]])
        Q = np.array([[0.01]])
        R = np.array([[0.01]])
        X_estimated = np.array([streamflow_data[0]])
        P_estimated = np.eye(1) * 0.01
        estimated_states = []

        for measurement in streamflow_data:
            # predict
            X_predicted = A.dot(X_estimated)
            P_predicted = A.dot(P_estimated).dot(A.T) + Q

            # update
            measurement_residual = measurement - H.dot(X_predicted)
            S = H.dot(P_predicted).dot(H.T) + R
            K = P_predicted.dot(H.T).dot(np.linalg.inv(S))  # kalman gain
            X_estimated = X_predicted + K.dot(measurement_residual)
            P_estimated = P_predicted - K.dot(H).dot(P_predicted)
            estimated_states.append(X_estimated.item())

        estimated_states = np.array(estimated_states)

        # Apply non-negative constraints
        estimated_states[estimated_states < 0] = 0
        return self.data_balanced(streamflow_data, estimated_states)

    def moving_average_difference(self, streamflow_data, window_size=20):
        """
        对流量数据应用滑动平均差算法进行平滑处理，并保持流量总量平衡。
        :window_size: 滑动窗口的大小
        """
        streamflow_data_series = pd.Series(streamflow_data)
        # Calculate the forward moving average（MU）
        forward_ma = streamflow_data_series.rolling(
            window=window_size, min_periods=1
        ).mean()

        # Calculate the backward moving average（MD）
        backward_ma = (
            streamflow_data_series.iloc[::-1]
            .rolling(window=window_size, min_periods=1)
            .mean()
            .iloc[::-1]
        )

        # Calculate the difference between the forward and backward sliding averages
        ma_difference = abs(forward_ma - backward_ma)

        # Apply non-negative constraints
        ma_difference[ma_difference < 0] = 0
        return self.data_balanced(streamflow_data, ma_difference.to_numpy())

    def quadratic_function(self, x, a, b, c):
        return a * x**2 + b * x + c

    def robust_fitting(self, streamflow_data, k=1.5):
        """
        对流量数据应用抗差修正算法进行平滑处理，并保持流量总量平衡。
        默认采用二次曲线进行拟合优化，该算法处理性能较差
        """
        time_steps = np.arange(len(streamflow_data))
        params, _ = curve_fit(self.quadratic_function, time_steps, streamflow_data)
        smoothed_streamflow = self.quadratic_function(time_steps, *params)
        residuals = streamflow_data - smoothed_streamflow
        m = len(streamflow_data)
        sigma = np.sqrt(np.sum(residuals**2) / (m - 1))

        for _ in range(10):
            weights = np.where(
                np.abs(residuals) <= k * sigma, 1, k * sigma / np.abs(residuals)
            )
            sigma = np.sqrt(np.sum(weights * residuals**2) / (m - 1))

        corrected_streamflow = (
            weights * streamflow_data + (1 - weights) * smoothed_streamflow
        )
        corrected_streamflow[corrected_streamflow < 0] = 0
        return self.data_balanced(streamflow_data, corrected_streamflow)

    def lowpass_filter(self, streamflow_data):
        """
        对一维流量数据应用调整后的低通滤波器。
        :cutoff_frequency: 低通滤波器的截止频率。
        :sampling_rate: 数据的采样率。
        :order: 滤波器的阶数，默认为5。
        """

        def apply_low_pass_filter(signal, cutoff_frequency, sampling_rate, order=5):
            nyquist_frequency = 0.5 * sampling_rate
            normalized_cutoff = cutoff_frequency / nyquist_frequency
            b, a = butter(order, normalized_cutoff, btype="low", analog=False)
            filtered_signal = filtfilt(b, a, signal)
            return filtered_signal

        # Apply a low-pass filter
        low_pass_filtered_signal = apply_low_pass_filter(
            streamflow_data, self.cutoff_frequency, self.sampling_rate, self.order
        )

        # Apply non-negative constraints
        low_pass_filtered_signal[low_pass_filtered_signal < 0] = 0

        return self.data_balanced(streamflow_data, low_pass_filtered_signal)

    def FFT(self, streamflow_data):
        """
        对流量数据进行迭代的傅里叶滤波处理，包括非负值调整和流量总量调整。
        :cutoff_frequency: 傅里叶滤波的截止频率。
        :time_step: 数据采样间隔。
        :iterations: 迭代次数。
        """
        current_signal = streamflow_data.to_numpy().copy()

        for _ in range(self.iterations):
            n = len(current_signal)
            yf = fft(current_signal)
            xf = fftfreq(n, d=self.time_step)

            # Applied frequency filtering
            yf[np.abs(xf) > self.cutoff_frequency] = 0

            # FFT and take the real part
            filtered_signal = ifft(yf).real

            # Apply non-negative constraints
            filtered_signal[filtered_signal < 0] = 0

            # Adjust the total flow to match the original flow
            current_signal = self.data_balanced(streamflow_data, filtered_signal)

        return current_signal

    def wavelet(self, streamflow_data):
        """
        对一维流量数据进行小波变换分析前后拓展数据以减少边缘失真，然后调整总流量。
        :cwt_row: 小波变换中使用的特定宽度。
        """
        streamflow_data_array = streamflow_data.to_numpy().copy()
        # Expand the data edge by 24 lines on each side
        extended_data = np.concatenate(
            [
                np.full(
                    24, streamflow_data_array[0]
                ),  # Expand the first 24 lines with the first element
                streamflow_data,
                np.full(
                    24, streamflow_data_array[-1]
                ),  # Expand the last 24 lines with the last element
            ]
        )
        widths = np.arange(1, 31)
        # Wavelet transform by Morlet wavelet directly
        extended_cwt = cwt(extended_data, morlet, widths)
        scaled_cwtmatr = np.abs(extended_cwt)

        # Select a specific width for analysis (can be briefly understood as selecting a cutoff frequency)
        cwt_row_extended = scaled_cwtmatr[self.cwt_row, :]

        # Remove the extended part
        adjusted_cwt_row = cwt_row_extended[24:-24]
        adjusted_cwt_row[adjusted_cwt_row < 0] = 0
        return self.data_balanced(streamflow_data, adjusted_cwt_row)

    def anomaly_process(self, methods=None):
        super().anomaly_process(methods)
        streamflow_data = self.origin_df["INQ"]
        for method in methods:
            if method == "moving_average":
                streamflow_data = self.moving_average(streamflow_data=streamflow_data)
            elif method == "kalman":
                streamflow_data = self.kalman_filter(streamflow_data=streamflow_data)
            elif method == "moving_average_diff":
                streamflow_data = self.moving_average_difference(
                    streamflow_data=streamflow_data
                )
            elif method == "robfit":
                streamflow_data = self.robust_fitting(streamflow_data=streamflow_data)
            elif method == "lowpass":
                streamflow_data = self.lowpass_filter(streamflow_data=streamflow_data)
            elif method == "FFT":
                streamflow_data = self.FFT(streamflow_data=streamflow_data)
            elif method == "wavelet":
                streamflow_data = self.wavelet(streamflow_data=streamflow_data)
            else:
                print("please check your method name")

        # self.processed_df["streamflow"] = streamflow_data  # 最终结果赋值给processed_df
        # 新增一列进行存储
        self.processed_df[str(methods)] = streamflow_data
