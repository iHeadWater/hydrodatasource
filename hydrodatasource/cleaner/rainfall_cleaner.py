
from .cleaner import Cleaner

class RainfallCleaner(Cleaner):
    def aggregate_data(self):
        # 实现聚合降雨数据的方法
        pass

    def identify_trends(self):
        # 实现识别降雨趋势的方法
        pass

    def anomaly_process(self, methods=None):
        super().process_data(methods)
        if "aggregate" in methods:
            self.aggregate_data()
        if "trends" in methods:
            self.identify_trends()
