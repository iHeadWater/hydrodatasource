import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import cwt, morlet, butter, filtfilt
from scipy.fft import fft, ifft, fftfreq
from scipy.optimize import curve_fit

def data_balanced(origin_data,transform_data):
    # Calculate the flow balance factor and keep the total volume consistent
    streamflow_data_before = np.sum(origin_data)
    streamflow_data_after = np.sum(transform_data)
    scaling_factor = streamflow_data_before / streamflow_data_after
    balanced_data = transform_data * scaling_factor

    print(f"Total flow (before smoothing): {streamflow_data_before}")
    print(f"Total flow (after smoothing): {np.sum(balanced_data)}")
    # print(f"scaling factor: {scaling_factor}")
    return balanced_data

def moving_average(streamflow_data, window_size=20):
    """
    对流量数据应用滑动平均进行平滑处理，并保持流量总量平衡。
    :window_size: 滑动窗口大小
    :return: 平滑处理后的流量数据
    """
    smoothed_data = np.convolve(streamflow_data, np.ones(window_size)/window_size, mode='same')
    
    # Apply non-negative constraints
    smoothed_data[smoothed_data < 0] = 0
    return data_balanced(streamflow_data,smoothed_data)

def kalman_filter(streamflow_data):
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
    return data_balanced(streamflow_data,estimated_states)

def moving_average_difference(streamflow_data, window_size=20):
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
    return data_balanced(streamflow_data,ma_difference)

def quadratic_function(x, a, b, c):
    return a * x**2 + b * x + c
def robust_fitting(streamflow_data, k=1.5):
    """
    对流量数据应用抗差修正算法进行平滑处理，并保持流量总量平衡。
    默认采用二次曲线进行拟合优化，该算法处理性能较差
    """
    time_steps = np.arange(len(streamflow_data))
    params, _ = curve_fit(quadratic_function, time_steps, streamflow_data)
    smoothed_streamflow = quadratic_function(time_steps, *params)
    residuals = streamflow_data - smoothed_streamflow
    m = len(streamflow_data)
    sigma = np.sqrt(np.sum(residuals**2) / (m - 1))

    for _ in range(10):
        weights = np.where(np.abs(residuals) <= k * sigma, 1, k * sigma / np.abs(residuals))
        sigma = np.sqrt(np.sum(weights * residuals**2) / (m - 1))

    corrected_streamflow = weights * streamflow_data + (1 - weights) * smoothed_streamflow
    corrected_streamflow[corrected_streamflow < 0] = 0
    return data_balanced(streamflow_data, corrected_streamflow)

def lowpass_filter(streamflow_data, cutoff_frequency, sampling_rate, order=5):
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
    low_pass_filtered_signal = apply_low_pass_filter(streamflow_data, cutoff_frequency, sampling_rate, order)
    
    # Apply non-negative constraints
    low_pass_filtered_signal[low_pass_filtered_signal < 0] = 0

    return data_balanced(streamflow_data, low_pass_filtered_signal)

def FFT(streamflow_data, cutoff_frequency=0.1, time_step=1.0, iterations=3):
    """
    对流量数据进行迭代的傅里叶滤波处理，包括非负值调整和流量总量调整。
    :cutoff_frequency: 傅里叶滤波的截止频率。
    :time_step: 数据采样间隔。
    :iterations: 迭代次数。
    """
    current_signal = streamflow_data.copy()

    for _ in range(iterations):
        n = len(current_signal)
        yf = fft(current_signal)
        xf = fftfreq(n, d=time_step)
        
        # Applied frequency filtering
        yf[np.abs(xf) > cutoff_frequency] = 0
        
        # FFT and take the real part
        filtered_signal = ifft(yf).real
        
        # Apply non-negative constraints
        filtered_signal[filtered_signal < 0] = 0
        
        # Adjust the total flow to match the original flow
        current_signal = data_balanced(streamflow_data, filtered_signal)

    return current_signal

def wavelet(streamflow_data, cwt_row=1):
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
    cwt_row_extended = scaled_cwtmatr[cwt_row, :]  
    
    # Remove the extended part
    adjusted_cwt_row = cwt_row_extended[24:-24]  
    adjusted_cwt_row[adjusted_cwt_row < 0] = 0
    return data_balanced(streamflow_data, adjusted_cwt_row)


def streamflow_smooth(file_path="/home/user/path/to/***.csv", column_flow='INQ', column_time='Time', start_time=None, end_time=None,preprocess=True, method='moving_average', save_path=None, plot=False, window_size=5, cutoff_frequency=0.1, sampling_rate=1.0, order=5,cwt_row=1):
    """
    Runoff Hyperparameters Definition
    ---------------------------------

    file_path: str
        Path to the data file.

    column_flow: str
        The column name for flow data in the DataFrame, e.g., df['INQ'].

    column_time: str
        The column name for time data in the DataFrame, e.g., df['Time'].

    start_time: str, optional
        The start time for the data range to be processed. If not specified, the entire dataset is used. Format should match the time format in the dataset.

    end_time: str, optional
        The end time for the data range to be processed. If not specified, the entire dataset is used. Format should match the time format in the dataset.

    preprocess: bool, optional
        Indicates whether to perform data preprocessing optimization (e.g., smoothing missing values). Defaults to True.

    method: str, optional
        The specific processing function type to be used. Options include 'FFT', 'wavelet', 'kalman', 'moving_average', 'moving_average_difference', 'robust', 'lowpass'. Defaults to 'moving_average'.

    window_size: int, optional
        The size of the moving window for the selected method. Defaults to 5.

    cutoff_frequency: float, optional
        The cutoff frequency for lowpass filtering, applicable if a filtering method is chosen. Defaults to 0.1.

    sampling_rate: float, optional
        The sampling rate of the data, relevant for certain processing methods. Defaults to 1.0.

    order: int, optional
        The order of the filter or other relevant parameter for the chosen method. Defaults to 5.

    cwt_row: int, optional
        The row parameter for the Continuous Wavelet Transform, if applicable. Defaults to 1.

    save_path: str, optional
        The path where the processed data should be saved. If not specified, the user must define how to store the results (e.g., creating a new file with ***.csv).

    plot: bool, optional
        Indicates whether to plot a graph of the processed data. Defaults to False, meaning no plot will be generated unless explicitly requested.

    Note:
        - The 'column_flow' and 'column_time' parameters expect column names as they appear in the DataFrame.
        - For 'start_time' and 'end_time', the format should match the time format in the dataset to ensure correct parsing.
        - The 'method' parameter choice determines which additional hyperparameters ('window_size', 'cutoff_frequency', etc.) are relevant and should be configured accordingly.

    """
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.nc'):
        ds = xr.open_dataset(file_path)
        df = ds.to_dataframe().reset_index()
        pass
    else:
        raise ValueError("Unsupported file format")
    
    df[column_time] = pd.to_datetime(df[column_time])
    df[column_flow] = pd.to_numeric(df[column_flow], errors='coerce')

    # Filters data for a specified time range
    min_time = df[column_time].min()
    max_time = df[column_time].max()

    # adjust start_time
    if start_time is None or pd.to_datetime(start_time) < min_time:
        start_time = min_time
    else:
        start_time = pd.to_datetime(start_time)

    # adjust end_time
    if end_time is None or pd.to_datetime(end_time) > max_time:
        end_time = max_time
    else:
        end_time = pd.to_datetime(end_time)

    # An empty DataFrame is returned If the adjusted start time is greater than the end time 
    if start_time > end_time:
        raise ValueError("Capture data time error")
    else:
        # Filter data based on adjusted start and end time
        df = df[(df[column_time] >= start_time) & (df[column_time] <= end_time)]

    
    #  Fill missing values as the average of the previous 10 hours
    if preprocess:
        df[column_flow] = df[column_flow].fillna(df[column_flow].rolling(window=11, min_periods=1).mean())
        # Populate the part that is still NaN with a fill value of 0
        df[column_flow] = df[column_flow].fillna(0)
    
    # Apply the selected method
    # extract One-dimensional flow data
    streamflow_data = df[column_flow].values.squeeze()
    if method == 'moving_average':
        filtered_data = moving_average(streamflow_data, window_size=window_size)
    elif method == 'FFT':
        filtered_data = FFT(streamflow_data, cutoff_frequency=cutoff_frequency, time_step=1.0, iterations=3)
    elif method == 'wavelet':
        filtered_data = wavelet(streamflow_data, cwt_row=cwt_row)
    elif method == 'kalman':
        filtered_data = kalman_filter(streamflow_data)
    elif method == 'lowpass':
        filtered_data = lowpass_filter(streamflow_data, cutoff_frequency=cutoff_frequency, sampling_rate=sampling_rate, order=order)
    elif method == 'moving_average_difference':
        filtered_data = moving_average_difference(streamflow_data, window_size=window_size)
    elif method == 'robust':
        filtered_data = robust_fitting(streamflow_data)   
    else:
        raise ValueError("Unsupported method")

    # save processed data
    if save_path:
        df[column_time] = pd.to_datetime(df[column_time])
        
        # Create a new column by populating NaN first
        df['Filtered_INQ'] = np.nan
        
        # Calibrate index range
        start_idx = df.index[df[column_time] >= pd.to_datetime(start_time)].min()
        end_idx = df.index[df[column_time] <= pd.to_datetime(end_time)].max()
        
        # Assume that filtered_data is the result of processing within the same start-stop time
        # Insert processed data into the corresponding position in the original data set
        df.loc[start_idx:end_idx, 'Filtered_INQ'] = filtered_data
        df.to_csv(save_path, index=False)

    
    # Draw a comparison image of data processing
    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(df[column_time], df[column_flow], label='Original')
        plt.plot(df[column_time], filtered_data, label='Filtered', linestyle='--')
        plt.legend()
        plt.xlabel('Time')
        plt.ylabel('Flow')
        plt.title('Streamflow Data Processing')
        plt.grid(True)
        plt.show()


# Function usage example
streamflow_smooth(file_path="/home/user/path/to/***.csv", column_flow='INQ', column_time='TM', start_time=None, end_time=None,preprocess=True, method='wavelet', save_path=None, plot=True, window_size=24, cutoff_frequency=0.01, sampling_rate=1.0, order=5,cwt_row=1)
