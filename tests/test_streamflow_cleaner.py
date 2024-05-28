'''
Author: liutiaxqabs 1498093445@qq.com
Date: 2024-04-22 13:38:07
LastEditors: liutiaxqabs 1498093445@qq.com
LastEditTime: 2024-05-28 14:39:24
FilePath: /hydrodatasource/tests/test_streamflow_cleaner.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import pytest
from hydrodatasource.cleaner.streamflow_cleaner import StreamflowCleaner  # 确保引入你的类
import pandas as pd
import matplotlib.pyplot as plt

def test_anomaly_process():
    # 测试径流数据处理功能
    cleaner = StreamflowCleaner("/home/liutianxv1/径流sampledatatest.csv", cwt_row=10)
    # methods默认可以联合调用，也可以单独调用。大多数情况下，默认调用moving_average
    methods = ["EMA"]
    cleaner.anomaly_process(methods)
    print(cleaner.origin_df)
    print(cleaner.processed_df)
    cleaner.processed_df.to_csv("/home/liutianxv1/径流sampledatatest.csv",index=False)

    # df = cleaner.processed_df[(cleaner.processed_df['TM']>'2020-07-01')&(cleaner.processed_df['TM']<'2020-10-01')]
    # df['TM'] = pd.to_datetime(df['TM'])
    # plt.plot(df['TM'], df['INQ'], label='INQ')  # 可以选择是否添加 marker
    # plt.plot(df['TM'], df[methods[0]], label=methods[0])  # 同样，可以选择是否添加 marker
    # # 添加图例
    # plt.legend()
    # # 添加标题和轴标签
    # plt.title('TM vs INQ and INQ_PRoCESS')
    # plt.xlabel('TM')
    # plt.ylabel('Streamflow')
    # # 设置x轴刻度
    # xticks = pd.date_range(start=df['TM'].min(), end=df['TM'].max(), freq='15D')
    # plt.xticks(xticks, [x.strftime('%Y-%m-%d') for x in xticks], rotation=45)  # 旋转45度，避免标签重叠

    # plt.savefig('/home/liutianxv1/plot.png')
    # # 显示图表
    # plt.show()
