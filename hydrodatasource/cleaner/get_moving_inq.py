import numpy as np
import pandas as pd
from pykalman import KalmanFilter


def calculate_esm(QFi, Qi):  # 计算平滑度
    numerator_list = []
    denominator_list = []
    for i in range(len(QFi) - 1):
        numerator = (QFi.values[i + 1] - QFi.values[i]) ** 2
        denominator = (Qi.values[i + 1] - Qi.values[i]) ** 2
        numerator_list.append(numerator)
        denominator_list.append(denominator)
    numerator_total = np.sum(numerator_list)
    denominator_total = np.sum(denominator_list)
    return 1 - numerator_total / denominator_total


def get_moving_average_inq(inq_data_df):
    """
    :param inq_data_df: 入库流量表格，需要有TM（时间）和INQ（入库流量）两列
    :return:
    """
    inq_data = inq_data_df["INQ"]
    inq_data_df["TM"] = pd.to_datetime(inq_data_df["TM"], format="%d/%m/%Y %H:%M")
    # 滑动平均
    window_size = 5
    inq_moving_average = np.convolve(
        inq_data_df["INQ"], np.ones(window_size) / window_size, mode="same"
    )
    # 五点三次
    QF = np.zeros(len(inq_data))
    QF[0] = (
        1
        / 70
        * (
            69 * inq_data[0]
            + 4 * inq_data[1]
            - 6 * inq_data[2]
            + 4 * inq_data[3]
            - inq_data[4]
        )
    )
    QF[1] = (
        1
        / 30
        * (
            2 * inq_data[0]
            + 27 * inq_data[1]
            + 12 * inq_data[2]
            - 8 * inq_data[3]
            + 2 * inq_data[4]
        )
    )
    for i in range(2, len(inq_data) - 2):
        QF[i] = (
            1
            / 35
            * (
                (-3) * inq_data[i - 2]
                + 12 * inq_data[i - 1]
                + 17 * inq_data[i]
                + 12 * inq_data[i + 1]
                - 3 * inq_data[i + 2]
            )
        )
    QF[len(inq_data) - 2] = (
        1
        / 35
        * (
            2 * inq_data[len(inq_data) - 5]
            - 8 * inq_data[len(inq_data) - 4]
            + 12 * inq_data[len(inq_data) - 3]
            + 27 * inq_data[len(inq_data) - 2]
            + 2 * inq_data[len(inq_data) - 1]
        )
    )
    QF[len(inq_data) - 1] = (
        1
        / 70
        * (
            (-1) * inq_data[len(inq_data) - 5]
            + 4 * inq_data[len(inq_data) - 4]
            - 6 * inq_data[len(inq_data) - 3]
            + 4 * inq_data[len(inq_data) - 2]
            + 69 * inq_data[len(inq_data) - 1]
        )
    )
    # 卡尔曼滤波
    initial_state_mean = 0
    initial_state_covariance = 0.5
    observation_covariance = 0.5
    transition_covariance = 0.5
    kf = KalmanFilter(
        initial_state_mean=initial_state_mean,
        initial_state_covariance=initial_state_covariance,
        observation_covariance=observation_covariance,
        transition_covariance=transition_covariance,
        n_dim_obs=1,
    )
    kf = kf.em(inq_data, n_iter=10)
    inq_Kalman, _ = kf.filter(inq_data)
    inq_Kalman = np.ravel(inq_Kalman)
    result_df = pd.DataFrame(
        {
            "TM": inq_data_df["TM"],
            "INQ_moving": inq_moving_average,
            "INQ_QF": QF,
            "INQ_Kalman": inq_Kalman.flatten(),
            "INQ_orig": inq_data_df["INQ"],
        }
    )
    # 计算误差
    W_bias_rela_moving_list = []
    Q_bias_rela_moving_list = []
    time_bias_moving_list = []
    esm_bias_moving_list = []
    W_bias_rela_Kalman_list = []
    Q_bias_rela_Kalman_list = []
    time_bias_Kalman_list = []
    esm_bias_Kalman_list = []
    W_bias_rela_QF_list = []
    Q_bias_rela_QF_list = []
    time_bias_QF_list = []
    esm_bias_QF_list = []
    inq_moving_filtered = result_df["INQ_moving"]
    inq_QF_filtered = result_df["INQ_QF"]
    inq_Kalman_filtered = result_df["INQ_Kalman"]
    inq_orig_filtered = result_df["INQ_orig"]
    # 滑动平均的误差
    W_bias_rela_moving = (
        abs(inq_moving_filtered.sum() - inq_orig_filtered.sum())
        / inq_orig_filtered.sum()
    )
    Q_bias_rela_moving = (
        abs(inq_moving_filtered.max() - inq_orig_filtered.max())
        / inq_orig_filtered.max()
    )
    time_bias_moving = (
        abs(inq_moving_filtered.argmax() - inq_orig_filtered.argmax())
        / inq_orig_filtered.argmax()
    )
    esm_bias_moving = calculate_esm(inq_moving_filtered, inq_orig_filtered)
    W_bias_rela_moving_list.append(W_bias_rela_moving)
    Q_bias_rela_moving_list.append(Q_bias_rela_moving)
    time_bias_moving_list.append(time_bias_moving)
    esm_bias_moving_list.append(esm_bias_moving)
    # 五点三次的误差
    W_bias_rela_QF = (
        abs(inq_QF_filtered.sum() - inq_orig_filtered.sum()) / inq_orig_filtered.sum()
    )
    Q_bias_rela_QF = (
        abs(inq_QF_filtered.max() - inq_orig_filtered.max()) / inq_orig_filtered.max()
    )
    time_bias_QF = (
        abs(inq_QF_filtered.argmax() - inq_orig_filtered.argmax())
        / inq_orig_filtered.argmax()
    )
    esm_bias_QF = calculate_esm(inq_QF_filtered, inq_orig_filtered)
    W_bias_rela_QF_list.append(W_bias_rela_QF)
    Q_bias_rela_QF_list.append(Q_bias_rela_QF)
    time_bias_QF_list.append(time_bias_QF)
    esm_bias_QF_list.append(esm_bias_QF)
    # 卡尔曼滤波的误差
    W_bias_rela_Kalman = (
        abs(inq_Kalman_filtered.sum() - inq_orig_filtered.sum())
        / inq_orig_filtered.sum()
    )
    Q_bias_rela_Kalman = (
        abs(inq_Kalman_filtered.max() - inq_orig_filtered.max())
        / inq_orig_filtered.max()
    )
    time_bias_Kalman = (
        abs(inq_Kalman_filtered.argmax() - inq_orig_filtered.argmax())
        / inq_orig_filtered.argmax()
    )
    esm_bias_Kalman = calculate_esm(inq_Kalman_filtered, inq_orig_filtered)
    W_bias_rela_Kalman_list.append(W_bias_rela_Kalman)
    Q_bias_rela_Kalman_list.append(Q_bias_rela_Kalman)
    time_bias_Kalman_list.append(time_bias_Kalman)
    esm_bias_Kalman_list.append(esm_bias_Kalman)
    inq_bias_df = pd.DataFrame(
        {
            "TM": inq_data_df["TM"],
            "INQ_moving": inq_moving_average,
            "INQ_QF": QF,
            "INQ_Kalman": inq_Kalman.flatten(),
            "INQ_orig": inq_data_df["INQ"],
        }
    )
    bias = pd.DataFrame(
        {
            # 'TM': inq_data_df['TM'],
            "W_bias_rela_moving_list": W_bias_rela_moving_list,
            "Q_bias_rela_moving_list": Q_bias_rela_moving_list,
            "time_bias_moving_list": time_bias_moving_list,
            "esm_bias_moving_list": esm_bias_moving_list,
            "W_bias_rela_Kalman_list": W_bias_rela_Kalman_list,
            "Q_bias_rela_Kalman_list": Q_bias_rela_Kalman_list,
            "time_bias_Kalman_list": time_bias_Kalman_list,
            "esm_bias_Kalman_list": esm_bias_Kalman_list,
            "W_bias_rela_QF_list": W_bias_rela_QF_list,
            "W_bias_rela_QF_list": W_bias_rela_QF_list,
            "Q_bias_rela_QF_list": Q_bias_rela_QF_list,
            "time_bias_QF_list": time_bias_QF_list,
            "esm_bias_QF_list": esm_bias_QF_list,
        }
    )
    return inq_bias_df
