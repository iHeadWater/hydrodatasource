"""
Author: liutiaxqabs 1498093445@qq.com
Date: 2024-04-19 14:00:27
LastEditors: liutiaxqabs 1498093445@qq.com
LastEditTime: 2024-04-19 14:03:40
FilePath: /liutianxv1/hydrodatasource/hydrodatasource/cleaner/waterlevel_cleaner.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
"""

from .cleaner import Cleaner
class WaterlevelCleaner(Cleaner):
    def detect_anomalies(self):
        # 实现检测水位异常的方法
        pass

    def normalize_data(self):
        # 实现正规化水位数据的方法
        pass

    def anomaly_process(self, methods=None):
        super().process_data(methods)
        if "detect_anomalies" in methods:
            self.detect_anomalies()
        if "normalize" in methods:
            self.normalize_data()
