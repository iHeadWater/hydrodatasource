import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import cwt, morlet, butter, filtfilt
from scipy.fft import fft, ifft, fftfreq
from scipy.optimize import curve_fit
import os

class Cleaner:
    """
    参数中文说明如下：
    - file_path (必选项): 文件路径，指定要处理的数据文件路径。支持.txt格式。
    - column_id: ID 列名，用于标识数据中的不同组或实例。若不表明则默认全体数据为一个分组。
    - ID_list: ID 列的值列表，指定要处理的特定 ID 列表。
    - column_flow (必选项): 流量数据列名，在 DataFrame 中表示流量的列名。
    - column_time (必选项): 时间数据列名，在 DataFrame 中表示时间的列名。
    - start_time: 起始时间，指定要处理的数据时间范围的起始时间。
    - end_time: 结束时间，指定要处理的数据时间范围的结束时间。
    - preprocess: 数据预处理标志，指示是否进行数据预处理优化（如平滑缺失值）。默认为 True。
    - method: 处理函数类型，要使用的特定处理函数类型。选项包括 'FFT'、'wavelet'、'kalman'、'moving_average'、'moving_average_difference'、'robust'、'lowpass'，默认为 'moving_average'。
    - save_path: 保存路径，指定处理后数据应保存的路径。如果未指定，则用户必须定义如何存储结果（例如创建一个新文件并命名为 ***.csv）。
    - plot: 绘图标志，指示是否绘制处理后数据的图形。默认为 False，表示不生成图形，除非明确请求。
    - window_size: 滑动窗口的大小，用于所选方法的滑动窗口大小。默认为 5。
    - cutoff_frequency: 低通滤波器的截止频率，适用于选择了滤波方法的情况。默认为 0.1。
    - sampling_rate: 数据的采样率，适用于某些处理方法。默认为 1.0。
    - order: 滤波器的阶数或其他相关参数，适用于所选方法的滤波器的阶数或其他相关参数。默认为 5。
    - cwt_row: 连续小波变换的行参数（如果适用）。默认为 1。
    - time_step: 时间步长，表示数据的时间步长。默认为 1.0。
    - iterations: 迭代次数，表示迭代的次数。默认为 3。
    """
    def __init__(self, file_path, column_id='ID', ID_list=None, column_flow='INQ', column_time='Time', start_time=None, end_time=None, preprocess=True, method='moving_average', save_path=None, plot=True, window_size=5, cutoff_frequency=0.1, time_step=1.0, iterations=3, sampling_rate=1.0, order=5, cwt_row=1):
        self.file_path = file_path
        self.column_id = column_id
        self.ID_list = ID_list
        self.column_flow = column_flow
        self.column_time = column_time
        self.start_time = start_time
        self.end_time = end_time
        self.preprocess = preprocess
        self.method = method
        self.save_path = save_path
        self.plot = plot
        self.window_size = window_size
        self.cutoff_frequency = cutoff_frequency
        self.sampling_rate = sampling_rate
        self.order = order
        self.cwt_row = cwt_row
        self.time_step = time_step
        self.iterations = iterations

    def data_balanced(self, origin_data,transform_data):
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
        smoothed_data = np.convolve(streamflow_data, np.ones(window_size)/window_size, mode='same')
        
        # Apply non-negative constraints
        smoothed_data[smoothed_data < 0] = 0
        return self.data_balanced(streamflow_data,smoothed_data)

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
        return self.data_balanced(streamflow_data,estimated_states)

    def moving_average_difference(self, streamflow_data, window_size=20):
        """
        对流量数据应用滑动平均差算法进行平滑处理，并保持流量总量平衡。
        :window_size: 滑动窗口的大小
        """
        streamflow_data_series = pd.Series(streamflow_data)
        # Calculate the forward moving average（MU）
        forward_ma = streamflow_data_series.rolling(window=window_size, min_periods=1).mean()

        # Calculate the backward moving average（MD）
        backward_ma = streamflow_data_series.iloc[::-1].rolling(window=window_size, min_periods=1).mean().iloc[::-1]

        # Calculate the difference between the forward and backward sliding averages
        ma_difference = abs(forward_ma - backward_ma)

        # Apply non-negative constraints
        ma_difference[ma_difference < 0] = 0
        return self.data_balanced(streamflow_data,ma_difference.to_numpy())
    

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
            weights = np.where(np.abs(residuals) <= k * sigma, 1, k * sigma / np.abs(residuals))
            sigma = np.sqrt(np.sum(weights * residuals**2) / (m - 1))

        corrected_streamflow = weights * streamflow_data + (1 - weights) * smoothed_streamflow
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
            b, a = butter(order, normalized_cutoff, btype='low', analog=False)
            filtered_signal = filtfilt(b, a, signal)
            return filtered_signal
        
        # Apply a low-pass filter
        low_pass_filtered_signal = apply_low_pass_filter(streamflow_data, self.cutoff_frequency, self.sampling_rate, self.order)
        
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
        current_signal = streamflow_data.copy()

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
        # Expand the data edge by 24 lines on each side
        extended_data = np.concatenate([
            np.full(24, streamflow_data[0]),  # Expand the first 24 lines with the first element
            streamflow_data,  
            np.full(24, streamflow_data[-1])  # Expand the last 24 lines with the last element
        ])
        widths=np.arange(1, 31)
        # Wavelet transform by Morlet wavelet directly
        extended_cwt = cwt(extended_data, morlet, widths)
        scaled_cwtmatr = np.abs(extended_cwt)
        
        # Select a specific width for analysis (can be briefly understood as selecting a cutoff frequency)
        cwt_row_extended = scaled_cwtmatr[self.cwt_row, :]  
        
        # Remove the extended part
        adjusted_cwt_row = cwt_row_extended[24:-24]  
        adjusted_cwt_row[adjusted_cwt_row < 0] = 0
        return self.data_balanced(streamflow_data, adjusted_cwt_row)


    def streamflow_smooth(self, df):
        #  Fill missing values as the average of the previous 10 hours
        if self.preprocess:
            df[self.column_flow] = df[self.column_flow].fillna(df[self.column_flow].rolling(window=11, min_periods=1).mean())
            # Populate the part that is still NaN with a fill value of 0
            df[self.column_flow] = df[self.column_flow].fillna(0)
        #Record the generated data
        df[self.column_time] = pd.to_datetime(df[self.column_time])

        # Calibrate index range
        start_idx = df.index[df[self.column_time] >= pd.to_datetime(self.start_time)].min()
        end_idx = df.index[df[self.column_time] <= pd.to_datetime(self.end_time)].max()
        
        # Apply the selected method
        # extract One-dimensional flow data
        streamflow_data = df.loc[start_idx:end_idx, self.column_flow]
        streamflow_data = streamflow_data.values.squeeze()
        print(streamflow_data)

        if self.method == 'moving_average':
            filtered_data = self.moving_average(streamflow_data)
        elif self.method == 'FFT':
            filtered_data = self.FFT(streamflow_data)
        elif self.method == 'wavelet':
            filtered_data = self.wavelet(streamflow_data)
        elif self.method == 'kalman':
            filtered_data = self.kalman_filter(streamflow_data)
        elif self.method == 'lowpass':
            filtered_data = self.lowpass_filter(streamflow_data)
        elif self.method == 'moving_average_difference':
            filtered_data = self.moving_average_difference(streamflow_data)
        elif self.method == 'robust':
            filtered_data = self.robust_fitting(streamflow_data)   
        else:
            raise ValueError("Unsupported method")
        
        # Assume that filtered_data is the result of processing within the same start-stop time
        # Insert processed data into the corresponding position in the original data set
        df.loc[start_idx:end_idx, self.method +'_INQ_Filtered'] = filtered_data
        
        return df

    def time_filter(self, df):
        # filter by watershed number + start and end time
        df[self.column_time] = pd.to_datetime(df[self.column_time])
        # Filters data for a specified time range
        min_time = df[self.column_time].min()
        max_time = df[self.column_time].max()

        # adjust start_time
        if self.start_time is None or pd.to_datetime(self.start_time) < min_time:
            self.start_time = min_time
        else:
            self.start_time = pd.to_datetime(self.start_time)

        # adjust end_time
        if self.end_time is None or pd.to_datetime(self.end_time) > max_time:
            self.end_time = max_time
        else:
            self.end_time = pd.to_datetime(self.end_time)

        # An empty DataFrame is returned If the adjusted start time is greater than the end time 
        if self.start_time > self.end_time:
            raise ValueError("Capture data time error")
        return self.start_time, self.end_time

    def drawplot(self, df, id):
        df = df[(df[self.column_time] > self.start_time) & (df[self.column_time] < self.end_time)]
        plt.figure(figsize=(10, 6))
        plt.plot(df[self.column_time], df[self.column_flow], label='Original')
        plt.plot(df[self.column_time], df[self.method +'_INQ_Filtered'], label='Filtered', linestyle='--')
        plt.legend()
        plt.xlabel('Time')
        plt.ylabel('Flow')
        if id:
            plt.title(f"BASIN ID: {id} Streamflow Data Processing")
        else:
            plt.title(f"All Dataset Streamflow Data Processing")
        plt.gcf().autofmt_xdate()  # Automatically rotate date markers to prevent overlapping
        plt.grid(True)
        plt.show()

    def process_inq(self):
        if self.file_path.endswith('.csv'):
            df = pd.read_csv(self.file_path, dtype={self.column_id: str})    

        elif self.file_path.endswith(('.nc', '.h5', '.netcdf')):
            ds = xr.open_dataset(self.file_path)
            # turn xarray into DataFrame
            df = ds.to_dataframe()            

        elif self.file_path.endswith('.txt'):
            # Assuming .txt files are comma-delimited, if they are tab-delimited, use sep='\t'
            df = pd.read_csv(self.file_path, sep=',', dtype={self.column_id: str})  

        else:
            raise ValueError("Unsupported file format")
        
        # data processing
        print(df)
        df = df.reset_index(drop=True)
        original_df = df
        df[self.column_flow] = pd.to_numeric(df[self.column_flow], errors='coerce')

        # time_filter
        df[self.column_time] = pd.to_datetime(df[self.column_time])
        self.start_time, self.end_time = self.time_filter(df)

        # add a new column and initialize as NaN if no exist
        if (self.method + '_INQ_Filtered') not in df.columns:
            df[self.method + '_INQ_Filtered'] = np.nan

        # group by-basin  
        if self.column_id is not None:  
            grouped = df.groupby(self.column_id)
            filtered_groups = []
            for id, group in grouped:
                try:
                    if self.ID_list is None or str(id) in self.ID_list:
                        print("当前处理数据为：")
                        print(group)
                        filtered_group = self.streamflow_smooth(group)
                        filtered_groups.append(filtered_group)

                        # Draw a comparison image of data processing
                        if self.plot:
                            self.drawplot(df=filtered_group, id=id)

                except Exception as e:
                    print(f"处理流域编号: {id} 的数据时发生错误: {e}")

            # Merge the filtered grouped data into a new DataFrame
            filtered_df = pd.concat(filtered_groups) 
        else:
            print("当前处理数据为整个数据集：")
            print(df)  # print all DataFrame
            filtered_df = self.streamflow_smooth(df)

            # Draw a comparison image of data processing
            if self.plot:
                self.drawplot(df=filtered_df, id=None)

        #save data
        if self.save_path:
            if self.file_path.endswith(('.nc', '.h5', '.netcdf')):
                # the 'basin' column should be an eight-digit string
                original_df[self.column_id] = original_df[self.column_id].astype(str).str.zfill(8)

            original_df[self.method + '_INQ_Filtered'] = filtered_df[self.method + '_INQ_Filtered']
            original_df.to_csv(self.save_path, index=False)
        
        return filtered_df