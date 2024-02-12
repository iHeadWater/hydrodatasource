import json

class DataConfig:
    def __init__(self):
        self.config = {
            # 流域编号，需要加区号
            "basin_id": "1_02051500",
            
            # 是否从hydro_opendata读取GPM_GFS文件
            "GPM_GFS_local_read": False,
            "GPM_GFS_merge": True,
            
            # 如果从本地读，需要给定GPM_GFS的路径,upload和local代表是否上传到Minio和是否下载到本地
            "GPM_GFS_local_path": "/home/xushuolong1/hydro_opendata/data",
            "GPM_GFS_upload": False,
            "GPM_GFS_local_save": True,
            
            # 是否从本地读取GPM/GFS文件
            "GPM_local_read": True,
            "GFS_local_read": True,
            
            # 是否需要从源数据合并新数据
            "GPM_merge": True,
            "GFS_merge": True,
            
            # 如果从本地读，需要给定GPM或者GFS的路径
            "GPM_local_path": "/home/xushuolong1/hydro_opendata/data/gpm.nc",
            "GFS_local_path": "/home/xushuolong1/hydro_opendata/data/gfs.nc",
            
            # 如果从MINIO读，需要给定MINIO的accesss_key和secret_key
            "MINIO_access_key": "",
            "MINIO_secret_key": "",
            
            # 数据类型，目前只有降水
            "data_type": "tp",
            
            # 拼接后的gpm_gfs的step中，GPM数据占据前多少个step
            "GPM_length": 169,
            
            # 拼接后的gpm_gfs的step中，GFS数据占据后多少个step
            "GFS_length": 23,
            
            # 拼接后的gpm_gfs的time_now，对应step中的第多少个值
            "time_now": 168,
            
            # 拼接后的gpm_gfs的time_now的时间范围
            "time_periods": [
                ["2017-01-10T00:00:00", "2017-01-11T00:00:00"],
                ["2017-01-15T00:00:00", "2017-01-20T00:00:00"],
            ],
            
            # 流域是国内还是国外，国外是camels，国内是wis
            "dataset": "camels",
            
            # 是否从hydro_opendata重新读取并生成gpm.nc，是否上传，是否保存在本地，如果保存，那么会保存至"GPM_local_path"
            # GFS基本同理，额外会有两个参数
            # enlarge参数，是因为GPM和GFS拼接的时候，需要GFS比GPM大一圈才能插值，不然会出现-
            # new参数，这个是为了模拟实际预报，因为实际预报的时候，GFS的数据不会是每一个时刻最新的数据
            # 但考虑到有时候算其他数据不需要大一圈，因此这里写了个参数
            "GPM_upload": False,
            "GPM_local_save": True,
            "GFS_enlarge": True,
            "GFS_new": True,
            "GFS_upload": False,
            "GFS_local_save": True,
            
            # 如果重新读取并生成gpm.nc，且没有mask.nc，那么需要提供GPM_mask，GFS同理
            "GPM_mask": False,
            "GPM_mask_path": "/home/xushuolong1/hydro_opendata/data/mask",
            "GFS_mask": False,
            "GFS_mask_path": "/home/xushuolong1/hydro_opendata/data/mask",
            
            # 如果需要生成mask.nc，那么需要提供流域的shp文件
            "shp_path": "/home/xushuolong1/hydro_opendata/data/shp/1_02051500/1_02051500.shp",
            
            # 如果生成GPM.nc，那么需要提供时间范围，理论上最好是连续的，以防特殊需求或者特殊情况，这里还是写成了分段的，GFS同理
            "GPM_time_periods": [
                ["2017-01-01T00:00:00", "2017-01-31T00:00:00"],
            ],
            "GFS_time_periods": [
                ["2017-01-01T00:00:00", "2017-01-31T00:00:00"],
            ]
        }
        
    def get_config(self):
        return self.config
    
