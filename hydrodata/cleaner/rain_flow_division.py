
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import os

def movmean(X, n):
    ones = np.ones(X.shape)
    kernel = np.ones(n)
    return np.convolve(X, kernel, mode='same') / np.convolve(ones, kernel, mode='same')


def step1_step2_tr_and_fluctuations_timeseries(rain, flow, rain_min, max_window):
    """
    :param rain: 降雨量向量，单位mm/h，需注意与mm/day之间的单位转化
    :param flow: 径流量向量，单位m³/h，需注意与m³/day之间的单位转化
    :param rain_min: 最小降雨量阈值
    :param max_window: 场次划分最大窗口，决定场次长度
    """
    rain = rain.T
    flow = flow.T
    rain_int = np.nancumsum(rain)
    flow_int = np.nancumsum(flow)
    T = rain.size
    rain_mean = np.empty(((max_window - 1) // 2, T))
    flow_mean = np.empty(((max_window - 1) // 2, T))
    fluct_rain = np.empty(((max_window - 1) // 2, T))
    fluct_flow = np.empty(((max_window - 1) // 2, T))
    F_rain = np.empty((max_window - 1) // 2)
    F_flow = np.empty((max_window - 1) // 2)
    F_rain_flow = np.empty((max_window - 1) // 2)
    rho = np.empty((max_window - 1) // 2)
    for window in np.arange(3, max_window + 1, 2):
        int_index = int((window - 1) / 2 - 1)
        start_slice = int(window - 0.5 * (window - 1))
        dst_slice = int(T - 0.5 * (window - 1))
        # 新建一个循环体长度*数据长度的大数组
        rain_mean[int_index] = movmean(rain_int, window)
        flow_mean[int_index] = movmean(flow_int, window)
        fluct_rain[int_index] = rain_int - rain_mean[int_index, :]
        F_rain[int_index] = (1 / (T - window + 1)) * np.nansum(
            (fluct_rain[int_index, start_slice:dst_slice]) ** 2)
        fluct_flow[int_index, np.newaxis] = flow_int - flow_mean[int_index, :]
        F_flow[int_index] = (1 / (T - window + 1)) * np.nansum(
            (fluct_flow[int_index, start_slice:dst_slice]) ** 2)
        F_rain_flow[int_index] = (1 / (T - window + 1)) * np.nansum(
            (fluct_rain[int_index, start_slice:dst_slice]) * (
                fluct_flow[int_index, start_slice:dst_slice]))
        rho[int_index] = F_rain_flow[int_index] / (
                np.sqrt(F_rain[int_index]) * np.sqrt(F_flow[int_index]))
    pos_min = np.argmin(rho)
    Tr = pos_min + 1
    tol_fluct_rain = (rain_min / (2 * Tr + 1)) * Tr
    tol_fluct_flow = flow_int[-1] / 1e15
    fluct_rain[pos_min, np.fabs(fluct_rain[pos_min, :]) < tol_fluct_rain] = 0
    fluct_flow[pos_min, np.fabs(fluct_flow[pos_min, :]) < tol_fluct_flow] = 0
    fluct_rain_Tr = fluct_rain[pos_min, :]
    fluct_flow_Tr = fluct_flow[pos_min, :]
    fluct_bivariate_Tr = fluct_rain_Tr * fluct_flow_Tr
    fluct_bivariate_Tr[np.fabs(fluct_bivariate_Tr) < np.finfo(np.float64).eps] = 0  # 便于比较
    
    return Tr, fluct_rain_Tr, fluct_flow_Tr, fluct_bivariate_Tr


def step3_core_identification(fluct_bivariate_Tr):
    d = np.diff(fluct_bivariate_Tr, prepend=[0], append=[0])  # 计算相邻数值差分，为0代表两端点处于0区间
    d[np.fabs(d) < np.finfo(np.float64).eps] = 0  # 确保计算正确
    d = np.logical_not(d)  # 求0-1数组，为真代表为0区间
    d0 = np.logical_not(np.convolve(d, [1, 1], 'valid'))  # 对相邻元素做OR，代表原数组数值是否处于某一0区间，再取反表示取有效值
    valid = np.logical_or(fluct_bivariate_Tr, d0)  # 有效core
    d_ = np.diff(valid, prepend=[0], append=[0])  # 求差分方便取上下边沿
    beginning_core = np.argwhere(d_ == 1)  # 上边沿为begin
    end_core = np.argwhere(d_ == -1) - 1  # 下边沿为end
 
    return beginning_core, end_core
    

def step4_end_rain_events(beginning_core, end_core, rain, fluct_rain_Tr, rain_min):
    end_rain = end_core.copy()
    rain = rain.T
    for g in range(end_core.size):
        if end_core[g] + 2 < fluct_rain_Tr.size and \
                (np.fabs(fluct_rain_Tr[end_core[g] + 1]) < np.finfo(np.float64).eps and np.fabs(fluct_rain_Tr[end_core[g] + 2]) < np.finfo(np.float64).eps):
            # case 1&2
            if np.fabs(rain[end_core[g]]) < np.finfo(np.float64).eps:
                # case 1
                while end_rain[g] > beginning_core[g] and np.fabs(rain[end_rain[g]]) < np.finfo(np.float64).eps:
                    end_rain[g] = end_rain[g] - 1
            else:
                # case 2
                bound = beginning_core[g + 1] if g + 1 < beginning_core.size else rain.size
                while end_rain[g] < bound and rain[end_rain[g]] > rain_min:
                    end_rain[g] = end_rain[g] + 1
                end_rain[g] = end_rain[g] - 1  # 回到最后一个
        else:
            # case 3
            # 若在降水，先跳过
            while end_rain[g] >= beginning_core[g] and rain[end_rain[g]] > rain_min:
                end_rain[g] = end_rain[g] - 1
            while end_rain[g] >= beginning_core[g] and rain[end_rain[g]] < rain_min:
                end_rain[g] = end_rain[g] - 1
    return end_rain


def step5_beginning_rain_events(beginning_core, end_rain, rain, fluct_rain_Tr, rain_min):
    beginning_rain = beginning_core.copy()
    rain = rain.T
    for g in range(beginning_core.size):
        if beginning_core[g] - 2 >= 0 \
                and (np.fabs(fluct_rain_Tr[beginning_core[g] - 1]) < np.finfo(np.float64).eps and np.fabs(fluct_rain_Tr[beginning_core[g] - 2]) < np.finfo(
            np.float64).eps) \
                and np.fabs(rain[beginning_core[g]]) < np.finfo(np.float64).eps:
            # case 1
            while beginning_rain[g] < end_rain[g] and np.fabs(rain[beginning_rain[g]]) < np.finfo(np.float64).eps:
                beginning_rain[g] = beginning_rain[g] + 1
        else:
            # case 2&3
            bound = end_rain[g - 1] if g - 1 >= 0 else -1
            while beginning_rain[g] > bound and rain[beginning_rain[g]] > rain_min:
                beginning_rain[g] = beginning_rain[g] - 1
            beginning_rain[g] = beginning_rain[g] + 1  # 回到第一个
    return beginning_rain


def step6_checks_on_rain_events(beginning_rain, end_rain, rain, rain_min, beginning_core, end_core):
    rain = rain.T
    beginning_rain = beginning_rain.copy()
    end_rain = end_rain.copy()
    if beginning_rain[0] == 0:  # 掐头
        beginning_rain = beginning_rain[1:]
        end_rain = end_rain[1:]
        beginning_core = beginning_core[1:]
        end_core = end_core[1:]
    if end_rain[-1] == rain.size - 1:  # 去尾
        beginning_rain = beginning_rain[:-2]
        end_rain = end_rain[:-2]
        beginning_core = beginning_core[:-2]
        end_core = end_core[:-2]
    error_time_reversed = beginning_rain > end_rain
    error_wrong_delimiter = np.logical_or(rain[beginning_rain - 1] > rain_min, rain[end_rain + 1] > rain_min)
    beginning_rain[error_time_reversed] = -2
    beginning_rain[error_wrong_delimiter] = -2
    end_rain[error_time_reversed] = -2
    end_rain[error_wrong_delimiter] = -2
    beginning_core[error_time_reversed] = -2
    beginning_core[error_wrong_delimiter] = -2
    end_core[error_time_reversed] = -2
    end_core[error_wrong_delimiter] = -2
    beginning_rain = beginning_rain[beginning_rain != -2]
    end_rain = end_rain[end_rain != -2]
    beginning_core = beginning_core[beginning_core != -2]
    end_core = end_core[end_core != -2]
    return beginning_rain, end_rain, beginning_core, end_core


def step7_end_flow_events(end_rain_checked, beginning_core, end_core, rain, fluct_rain_Tr, fluct_flow_Tr, Tr):
    end_flow = np.empty(end_core.size, dtype=int)
    for g in range(end_rain_checked.size):
        if end_core[g] + 2 < fluct_rain_Tr.size and \
                (np.fabs(fluct_rain_Tr[end_core[g] + 1]) < np.finfo(np.float64).eps and np.fabs(fluct_rain_Tr[end_core[g] + 2]) < np.finfo(np.float64).eps):
            # case 1
            end_flow[g] = end_rain_checked[g]
            bound = beginning_core[g + 1] + Tr if g + 1 < beginning_core.size else rain.size
            bound = min(bound, rain.size)  # 防溢出
            # 若flow为负，先跳过
            while end_flow[g] < bound and fluct_flow_Tr[end_flow[g]] <= 0:
                end_flow[g] = end_flow[g] + 1
            while end_flow[g] < bound and fluct_flow_Tr[end_flow[g]] > 0:
                end_flow[g] = end_flow[g] + 1
            end_flow[g] = end_flow[g] - 1  # 回到最后一个
        else:
            # case 2
            end_flow[g] = end_core[g]
            while end_flow[g] >= beginning_core[g] and fluct_flow_Tr[end_flow[g]] <= 0:
                end_flow[g] = end_flow[g] - 1
    return end_flow


def step8_beginning_flow_events(beginning_rain_checked, end_rain_checked, rain, beginning_core, fluct_rain_Tr, fluct_flow_Tr):
    beginning_flow = np.empty(beginning_rain_checked.size, dtype=int)
    for g in range(beginning_rain_checked.size):
        if beginning_core[g] - 2 >= 0 \
                and (np.fabs(fluct_rain_Tr[beginning_core[g] - 1]) < np.finfo(np.float64).eps and np.fabs(fluct_rain_Tr[beginning_core[g] - 2]) < np.finfo(
            np.float64).eps):
            beginning_flow[g] = beginning_rain_checked[g]  # case 1
        else:
            beginning_flow[g] = beginning_core[g]  # case 2
        while beginning_flow[g] < end_rain_checked[g] and fluct_flow_Tr[beginning_flow[g]] >= 0:
            beginning_flow[g] = beginning_flow[g] + 1
    return beginning_flow


def step9_checks_on_flow_events(beginning_rain_checked, end_rain_checked, beginning_flow, end_flow, fluct_flow_Tr):
    beginning_flow_checked = np.empty_like(beginning_flow, dtype=object)
    end_flow_checked = np.empty_like(end_flow, dtype=object)
    for g in range(len(beginning_flow)):
        if not np.isnan(beginning_flow[g]) and not np.isnan(end_flow[g]) and \
        (end_flow[g] <= beginning_flow[g] or fluct_flow_Tr[beginning_flow[g]] > 0 or fluct_flow_Tr[end_flow[g]] < 0 or \
            beginning_flow[g] < beginning_rain_checked[g] or end_flow[g] < end_rain_checked[g]):
            beginning_flow_checked[g] = np.nan
            end_flow_checked[g] = np.nan
        else:
            beginning_flow_checked[g] = beginning_flow[g]
            end_flow_checked[g] = end_flow[g]

    index_events = [i for i in range(len(beginning_rain_checked)) if
                not np.isnan(beginning_rain_checked[i]) and
                not np.isnan(beginning_flow_checked[i]) and
                not np.isnan(end_rain_checked[i]) and
                not np.isnan(end_flow_checked[i])]

    beginning_flow_ungrouped = beginning_flow_checked[index_events]
    end_flow_ungrouped = end_flow_checked[index_events]
    beginning_rain_ungrouped = beginning_rain_checked[index_events]
    end_rain_ungrouped = end_rain_checked[index_events]
    return beginning_rain_ungrouped, end_rain_ungrouped,beginning_flow_ungrouped, end_flow_ungrouped    


def step10_checks_on_overlapping_events(beginning_rain_ungrouped, end_rain_ungrouped, beginning_flow_ungrouped, end_flow_ungrouped,time):
    
    # # rain
    # order1 = np.reshape(np.hstack((np.reshape(beginning_rain_ungrouped, (-1, 1)),
    #                                np.reshape(end_rain_ungrouped, (-1, 1)))), (1, -1))
    # reversed1 = np.diff(order1) <= 0
    # order1[np.hstack((reversed1, [[False]]))] = -2
    # order1[np.hstack(([[False]], reversed1))] = -2
    # order1 = order1[order1 != -2]
    # # flow
    # order2 = np.reshape(np.hstack((np.reshape(beginning_flow_ungrouped, (-1, 1)),
    #                                np.reshape(end_flow_ungrouped, (-1, 1)))), (1, -1))
    # reversed2 = np.diff(order2) <= 0
    # order2[np.hstack((reversed2, [[False]]))] = -3
    # order2[np.hstack(([[False]], reversed2))] = -3
    # order2 = order2[order2 != -3]
    # # group
    # rain_grouped = np.reshape(order1, (-1, 2)).T
    # beginning_rain_grouped = rain_grouped[0]
    # end_rain_grouped = rain_grouped[1]
    # flow_grouped = np.reshape(order2, (-1, 2)).T
    # beginning_flow_grouped = flow_grouped[0]
    # end_flow_grouped = flow_grouped[1]
    # beginning_rain_grouped = beginning_rain_grouped.astype(int)
    # end_rain_grouped= end_rain_grouped.astype(int)
    # beginning_flow_grouped = beginning_flow_grouped.astype(int)
    # end_flow_grouped= end_flow_grouped.astype(int)
    # return time[beginning_rain_grouped], time[end_rain_grouped], time[beginning_flow_grouped], time[end_flow_grouped]
    beginning_rain_ungrouped = beginning_rain_ungrouped.astype(float)
    beginning_flow_ungrouped = beginning_flow_ungrouped.astype(float)
    end_rain_ungrouped = end_rain_ungrouped.astype(float)
    end_flow_ungrouped = end_flow_ungrouped.astype(float)
    q = 1
    marker_overlapping = []
    for g in range(len(end_rain_ungrouped) - 1):
        if end_rain_ungrouped[g] > beginning_rain_ungrouped[g + 1] or end_flow_ungrouped[g] > beginning_flow_ungrouped[g + 1]:
            marker_overlapping.append(g)
            q += 1
    if marker_overlapping:
        q = 0
        while q < len(marker_overlapping)-1:
            to_group = [marker_overlapping[q]]
            while q < len(marker_overlapping) and marker_overlapping[q] == marker_overlapping[q + 1] - 1:
                to_group.append(marker_overlapping[q + 1])
                q += 1
            
            beginning_rain_ungrouped[to_group[0]] = beginning_rain_ungrouped[to_group[0]]
            beginning_flow_ungrouped[to_group[0]] = beginning_flow_ungrouped[to_group[0]]
            end_flow_ungrouped[to_group[0]] = end_flow_ungrouped[to_group[-1] + 1]
            end_rain_ungrouped[to_group[0]] = end_rain_ungrouped[to_group[-1] + 1]

            if len(to_group) > 1:
                beginning_rain_ungrouped[to_group[1:]] = np.nan
                beginning_flow_ungrouped[to_group[1:]] = np.nan
                end_flow_ungrouped[to_group[1:]] = np.nan
                end_rain_ungrouped[to_group[1:]] = np.nan

            beginning_rain_ungrouped[to_group[-1] + 1] = np.nan
            beginning_flow_ungrouped[to_group[-1] + 1] = np.nan
            end_flow_ungrouped[to_group[-1] + 1] = np.nan
            end_rain_ungrouped[to_group[-1] + 1] = np.nan

            to_group = []
            q += 1

    index_events2 = np.where(~np.isnan(beginning_rain_ungrouped) & ~np.isnan(beginning_flow_ungrouped) & ~np.isnan(end_rain_ungrouped) & ~np.isnan(end_flow_ungrouped))[0]
    beginning_flow_grouped = beginning_flow_ungrouped[index_events2]
    end_flow_grouped = end_flow_ungrouped[index_events2]
    beginning_rain_grouped = beginning_rain_ungrouped[index_events2]
    end_rain_grouped = end_rain_ungrouped[index_events2]
    beginning_rain_grouped = beginning_rain_grouped.astype(int)
    end_rain_grouped= end_rain_grouped.astype(int)
    beginning_flow_grouped = beginning_flow_grouped.astype(int)
    end_flow_grouped = end_flow_grouped.astype(int)
    BEGINNING_RAIN = time[beginning_rain_grouped]
    END_RAIN = time[end_rain_grouped]
    END_FLOW = time[end_flow_grouped]
    BEGINNING_FLOW = time[beginning_flow_grouped]
    return BEGINNING_RAIN, END_RAIN, BEGINNING_FLOW, END_FLOW

def BASEFLOW_CURVE(BEGINNING_FLOW, END_FLOW, flow, time):
    baseflow = np.copy(flow)
    beg_end_series = np.array([])
    for j in range(len(BEGINNING_FLOW)):
        beg_end_series = np.concatenate((beg_end_series, [BEGINNING_FLOW[j], END_FLOW[j]]))

    for k in range(len(beg_end_series) - 1):
        index_beg = np.where(time == beg_end_series[k])[0]
        index_end = np.where(time == beg_end_series[k + 1])[0]
        index_beg = int(index_beg)
        index_end = int(index_end)
        if len(np.where(np.isnan(flow[index_beg:index_end + 1]) == 1)[0]) >= len(flow[index_beg:index_end + 1]) * 0.9:
            baseflow[index_beg:index_end + 1] = np.nan
        elif index_end - index_beg == 1:
            baseflow[index_beg] = flow[index_beg]
            baseflow[index_end] = flow[index_end]
        elif flow[index_beg] < flow[index_end]:
            increment = (flow[index_end] - flow[index_beg]) / (index_end - index_beg)
            for m in range(index_beg + 1, index_end):
                baseflow[m] = baseflow[index_beg] + increment * (m - index_beg)
        elif flow[index_beg] > flow[index_end]:
            increment = (flow[index_beg] - flow[index_end]) / (index_end - index_beg)
            for m in range(index_beg + 1, index_end):
                baseflow[m] = baseflow[index_beg] - increment * (m - index_beg)

    for m in range(len(baseflow)):
        if baseflow[m] > flow[m]:
            baseflow[m] = flow[m]

    return baseflow


def event_analysis(BEGINNING_RAIN, END_RAIN, BEGINNING_FLOW, END_FLOW, rain, flow, time, flag, multiple,flow_max_mm_h,duration_max):
    DURATION_RAIN = np.zeros(len(BEGINNING_RAIN))
    DURATION_RUNOFF = np.zeros(len(BEGINNING_FLOW))
    VOLUME_RAIN = np.zeros(len(BEGINNING_RAIN))
    VOLUME_RUNOFF = np.zeros(len(BEGINNING_FLOW))
    RUNOFF_RATIO = np.zeros(len(BEGINNING_RAIN))
    for h in range(len(BEGINNING_RAIN)):
        DURATION_RAIN[h] = (np.datetime64(END_RAIN[h]) - np.datetime64(BEGINNING_RAIN[h])) / np.timedelta64(1, 's') / (60 * 60 * multiple)
        DURATION_RUNOFF[h] = (np.datetime64(END_FLOW[h]) - np.datetime64(BEGINNING_FLOW[h])) / np.timedelta64(1, 's') / (60 * 60 * multiple)
        index_beginning_event = np.where(time == BEGINNING_RAIN[h])[0]
        index_end_event = np.where(time == END_RAIN[h])[0]
        index_beginning_event = int(index_beginning_event)
        index_end_event = int(index_end_event)
        VOLUME_RAIN[h] = np.nansum(rain[index_beginning_event:index_end_event]) * multiple

    if flag == 1:
        for h in range(len(BEGINNING_FLOW)):
            index_beginning_event = np.where(time == BEGINNING_FLOW[h])[0]
            index_end_event = np.where(time == END_FLOW[h])[0]
            VOLUME_RUNOFF[h] = np.nansum(flow[index_beginning_event:index_end_event]) * multiple
    else:
        baseflow = BASEFLOW_CURVE(BEGINNING_FLOW, END_FLOW, flow, time)
        for h in range(len(BEGINNING_FLOW)):
            index_beginning_event = np.where(time == BEGINNING_FLOW[h])[0]
            index_end_event = np.where(time == END_FLOW[h])[0]
            index_beginning_event = int(index_beginning_event)
            index_end_event = int(index_end_event)
            
            q = flow[index_beginning_event:index_end_event]
            qb = baseflow[index_beginning_event:index_end_event]
            VOLUME_RUNOFF[h] = np.nansum(q - qb) * multiple
    RUNOFF_RATIO = VOLUME_RUNOFF / VOLUME_RAIN
    result_df = pd.DataFrame({
        'BEGINNING_RAIN': BEGINNING_RAIN, 
        'END_RAIN': END_RAIN, 
        'BEGINNING_FLOW': BEGINNING_FLOW,
        'END_FLOW': END_FLOW,
        'DURATION_RAIN': DURATION_RAIN,
        'DURATION_RUNOFF': DURATION_RUNOFF,
        'VOLUME_RAIN': VOLUME_RAIN,
        'VOLUME_RUNOFF': VOLUME_RUNOFF,
        'RUNOFF_RATIO': RUNOFF_RATIO})
    drop_list=[]
    for i in range(0,len(BEGINNING_RAIN)):
        index_beginning_event = np.where(time == BEGINNING_FLOW[i])[0]
        index_end_event = np.where(time == END_FLOW[i])[0]
        index_beginning_event = int(index_beginning_event)
        index_end_event = int(index_end_event)
        if flow[index_beginning_event:index_end_event].max() < flow_max_mm_h :
            drop_list.append(i)
        if result_df['DURATION_RAIN'].iloc[i] > duration_max:
            drop_list.append(i)
        if result_df['DURATION_RUNOFF'].iloc[i] > duration_max:
            drop_list.append(i)
        if result_df['VOLUME_RAIN'].iloc[i] == 0:
            drop_list.append(i)
        if result_df['VOLUME_RUNOFF'].iloc[i] == 0:
            drop_list.append(i)
        if RUNOFF_RATIO[i] > 1:
            drop_list.append(i)      
    drop_array = np.unique(np.array(drop_list, dtype=int))
    result_df = result_df.drop(index=drop_array)
    DURATION_RAIN = result_df['DURATION_RAIN']
    VOLUME_RAIN = result_df['VOLUME_RAIN']
    DURATION_RUNOFF = result_df['DURATION_RUNOFF']
    VOLUME_RUNOFF = result_df['VOLUME_RUNOFF']
    RUNOFF_RATIO = result_df['RUNOFF_RATIO']
    BEGINNING_RAIN = result_df['BEGINNING_RAIN']
    END_RAIN = result_df['END_RAIN']
    BEGINNING_FLOW = result_df['BEGINNING_FLOW']
    END_FLOW = result_df['END_FLOW']
    return DURATION_RAIN, VOLUME_RAIN, DURATION_RUNOFF, VOLUME_RUNOFF, RUNOFF_RATIO,BEGINNING_RAIN, END_RAIN, BEGINNING_FLOW, END_FLOW


class Dcma_esr:
    def __init__(self, rain_min=0.02, max_window=100, flow_max=100, duration_max=2400, multiple=1, biliu_area=2097,plot=True,
                 rain_file=None, flow_file=None, save_path=None):
        self.rain_min = rain_min
        self.max_window = max_window
        self.flow_max = flow_max
        self.duration_max = duration_max
        self.multiple = multiple
        self.biliu_area = biliu_area
        self.plot = plot
        self.rain_file = rain_file
        self.flow_file = flow_file
        self.save_path = save_path

    def draw_rain_flow_division(self):
        # 加载数据
        biliu_flow_division = pd.read_csv(self.save_path)
        filtered_rain_aver_df = pd.read_csv(self.rain_file, engine='c')
        inq_interpolated_df = pd.read_csv(self.flow_file, engine='c', parse_dates=['TM'])

        # 转换日期格式
        filtered_rain_aver_df['TM'] = pd.to_datetime(filtered_rain_aver_df['TM'], format="%Y-%m-%d %H:%M:%S")
        inq_interpolated_df['TM'] = pd.to_datetime(inq_interpolated_df['TM'], format="%Y-%m-%d %H:%M:%S")

        # 遍历每个分割事件
        for i in range(len(biliu_flow_division)):
            beginning_time = biliu_flow_division['BEGINNING_RAIN'][i]
            end_time = biliu_flow_division['END_FLOW'][i]

            # 获取对应时间段的数据
            filtered_rain_aver_data = filtered_rain_aver_df[(filtered_rain_aver_df['TM'] >= beginning_time) & (filtered_rain_aver_df['TM'] <= end_time)]
            inq_interpolated_data = inq_interpolated_df[(inq_interpolated_df['TM'] >= beginning_time) & (inq_interpolated_df['TM'] <= end_time)]

            # 绘制图表
            fig, ax = plt.subplots(figsize=(9, 6))
            ax.plot(
                filtered_rain_aver_data['TM'],
                inq_interpolated_data['INQ'],#INQ_bias
                color="blue",
                linestyle="-",
                linewidth=1,
                label="INQ_bias",
            )
            ylim = np.max(inq_interpolated_data['INQ'])
            ax.set_ylim(0, ylim * 1.3)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            ax.set_xlabel('日期')
            ax.set_ylabel('流量 (m³/s)')
            plt.legend(loc="upper right")

            ax2 = ax.twinx()
            ax2.bar(
                filtered_rain_aver_data['TM'],
                filtered_rain_aver_data['rain'],
                color="grey",
                width=0.02,
                label="Rain"
            )
            ax2.set_ylabel('降雨量 (mm/h)')
            plt.legend(loc="upper left")
            # 对 ax 应用时间格式化和标签倾斜
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            ax.tick_params(axis='x', rotation=45)  # 应用于 ax 的 x 轴标签
            plt.title("降雨流量分割事件图示")
            plt.show()


    def process_rain_flow_division(self):
        # rain和flow之间的索引要尽量“对齐”
        # 2014.1.1 00:00:00-2022.9.1 00:00:00
        filtered_rain_aver_df = pd.read_csv(self.rain_file, engine='c').set_index('TM')
        filtered_rain_aver_array = filtered_rain_aver_df['rain'].to_numpy()
        flow_mm_h_df= pd.read_csv(self.flow_file, engine='c').set_index('TM')
        flow =flow_mm_h_df['INQ']
        # biliu_area = gdf_biliu_shp.geometry[0].area * 12100
        biliu_area = self.biliu_area
        flow_mm_h= flow.apply(lambda x: x * 3.6 / biliu_area)
        time = filtered_rain_aver_df.index
        time = pd.to_datetime(time, format="%Y-%m-%d %H:%M:%S")
        rain_min = self.rain_min
        max_window = self.max_window
        Tr, fluct_rain_Tr, fluct_flow_Tr, fluct_bivariate_Tr = step1_step2_tr_and_fluctuations_timeseries(
            filtered_rain_aver_array, flow_mm_h,
            rain_min,
            max_window)

        beginning_core, end_core = step3_core_identification(fluct_bivariate_Tr)
        end_rain = step4_end_rain_events(beginning_core, end_core, filtered_rain_aver_array, fluct_rain_Tr, rain_min)
        beginning_rain = step5_beginning_rain_events(beginning_core, end_rain, filtered_rain_aver_array, fluct_rain_Tr,
                                                    rain_min)
        beginning_rain_checked, end_rain_checked, beginning_core, end_core = step6_checks_on_rain_events(beginning_rain,
                                                                                                        end_rain,
                                                                                                        filtered_rain_aver_array,
                                                                                                        rain_min,
                                                                                                        beginning_core,
                                                                                                        end_core)
        end_flow = step7_end_flow_events(end_rain_checked, beginning_core, end_core, filtered_rain_aver_array,
                                        fluct_rain_Tr, fluct_flow_Tr, Tr)
        beginning_flow = step8_beginning_flow_events(beginning_rain_checked, end_rain_checked, filtered_rain_aver_array,
                                                    beginning_core,
                                                    fluct_rain_Tr, fluct_flow_Tr)
        beginning_rain_ungrouped,end_rain_ungrouped,beginning_flow_ungrouped, end_flow_ungrouped = step9_checks_on_flow_events(beginning_rain_checked, end_rain_checked,
                                                                            beginning_flow,
                                                                            end_flow, fluct_flow_Tr)
        BEGINNING_RAIN, END_RAIN, BEGINNING_FLOW, END_FLOW = step10_checks_on_overlapping_events(beginning_rain_ungrouped,
                                                                                                end_rain_ungrouped,
                                                                                                beginning_flow_ungrouped,
                                                                                                end_flow_ungrouped,
                                                                                                time)
        print(len(BEGINNING_RAIN), len(END_RAIN), len(BEGINNING_FLOW), len(END_FLOW))
        # print('_________________________')
        # print('_________________________')
        # print(BEGINNING_FLOW, END_FLOW)
        multiple=self.multiple
        flag=0
        flow_max = self.flow_max
        biliu_area = self.biliu_area
        flow_max_mm_h = flow_max * 3.6 / biliu_area
        duration_max = self.duration_max
        DURATION_RAIN,VOLUME_RAIN, DURATION_RUNOFF,VULUME_RUNOFF,RUNOFF_RATIO,BEGINNING_RAIN,END_RAIN,BEGINNING_FLOW,END_FLOW= event_analysis(BEGINNING_RAIN, END_RAIN, BEGINNING_FLOW, END_FLOW, 
                                                                                                filtered_rain_aver_array,flow_mm_h,time, flag, multiple,flow_max_mm_h,duration_max)
        biliu_division =  pd.DataFrame ({'BEGINNING_RAIN': BEGINNING_RAIN, 'END_RAIN': END_RAIN,'DURATION_RAIN':DURATION_RAIN,'BEGINNING_FLOW': BEGINNING_FLOW , 'END_FLOW': END_FLOW ,'DURATION_RUNOFF':DURATION_RUNOFF,'VOLUME_RAIN':VOLUME_RAIN,'VULUME_RUNOFF':VULUME_RUNOFF,'RUNOFF_RATIO':RUNOFF_RATIO})

        biliu_division.to_csv(self.save_path, index=False)

        #绘制场次图像
        if self.plot:
            self.draw_rain_flow_division()
