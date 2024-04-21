from .cleaner import Cleaner
class StreamflowCleaner(Cleaner):
    def filter_noise(self):
        # 实现过滤流量数据中的噪声的方法
        pass

    def compute_statistics(self):
        # 实现计算流量数据的统计信息的方法
        pass

    def anomaly_process(self, methods=None):
        super().process_data(methods)
        if "filter_noise" in methods:
            self.filter_noise()
        if "statistics" in methods:
            self.compute_statistics()
