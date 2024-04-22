import pytest
from hydrodatasource.cleaner.streamflow_cleaner import StreamflowCleaner  # 确保引入你的类
import pandas as pd
import matplotlib.pyplot as plt

def test_anomaly_process():
    # 测试径流数据处理功能
    cleaner = StreamflowCleaner("/home/liutianxv1/径流sampledatatest.csv", cwt_row=10)
    # methods默认可以联合调用，也可以单独调用。大多数情况下，默认调用moving_average
    methods = ['wavelet']
    cleaner.anomaly_process(methods)
    print(cleaner.origin_df)
    print(cleaner.processed_df)
    cleaner.processed_df.to_csv("/home/liutianxv1/径流sampledatatest.csv")
    
    # df = cleaner.processed_df
    # df['TM'] = pd.to_datetime(df['TM'])  # 确保'TM'列是 datetime 类型
    # filtered_df = df[(df['TM'] >= '2019-07-01') & (df['TM'] <= '2019-09-30')]

    # # 绘图
    # plt.figure(figsize=(10, 6))
    # plt.plot(filtered_df['TM'], filtered_df['INQ'], label='INQ', color='blue')
    # plt.plot(filtered_df['TM'], filtered_df[str(methods)], label=str(methods), color='red')
    # plt.xlabel('Time')
    # plt.ylabel('Values')
    # plt.title('INQ and INQ_Filter')
    # plt.legend()
    # plt.grid(True)

    # # 保存图像
    # plt.savefig('/home/liutianxv1/plot.png')
    # plt.show()
