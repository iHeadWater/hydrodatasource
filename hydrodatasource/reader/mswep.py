"""
Author: Wenyu Ouyang
Date: 2024-01-03 15:40:58
LastEditTime: 2025-03-19 16:55:02
LastEditors: Wenyu Ouyang
Description: Reading MSWEP data
FilePath: \hydrodatasource\hydrodatasource\reader\mswep.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import os
from collections import OrderedDict
import glob
import pandas as pd
from pathlib import Path
import numpy as np
import xarray as xr
from tqdm import tqdm

from hydrodatasource.reader.data_source import HydroData


class Mswep(HydroData):
    """Reading MSWEP precipitation data."""

    def __init__(self, data_path):
        """初始化MSWEP数据读取器

        Parameters
        ----------
        data_path : str
            MSWEP数据的根目录路径
        """
        self.data_source_dir = data_path
        self.data_source_description = self.set_data_source_describe()

    def get_name(self):
        return "MSWEP"

    def set_data_source_describe(self):
        """设置数据源描述信息，包括数据目录结构"""
        data_root_dir = self.data_source_dir
        periods = ["NRT", "Past"]
        time_resolutions = ["Daily", "3hourly", "Monthly"]

        data_description = {}
        for period in periods:
            period_dir = os.path.join(data_root_dir, period)
            data_description[period] = {}
            for resolution in time_resolutions:
                res_dir = os.path.join(period_dir, resolution)
                data_description[period][resolution] = res_dir

        return OrderedDict(
            DATA_DIR=data_root_dir,
            PERIOD_DATA=data_description,
        )

    def check_data_source(self):
        """This is a one-time check for the integrity of the data source.
        Please run it when you first use the data source.

        Raises
        ------
        ValueError
            如果存在不可读文件或变量缺失
        """
        error_files = []
        period_data = self.data_source_description["PERIOD_DATA"]

        # 预扫描所有文件路径
        print("🕵 扫描文件目录结构...")
        all_nc_files = []
        for period in period_data:
            for time_res in period_data[period]:
                data_dir = period_data[period][time_res]
                if not os.path.isdir(data_dir):
                    continue
                nc_files = glob.glob(os.path.join(data_dir, "*.nc"))
                all_nc_files.extend(nc_files)

        # 创建进度条
        with tqdm(
            total=len(all_nc_files),
            unit="file",
            desc="🔍 检查数据文件",
            dynamic_ncols=True,
        ) as pbar:
            for nc_file in all_nc_files:
                try:
                    # 更新进度条描述
                    pbar.set_postfix(file=f"{nc_file}...")

                    # 检查文件可读性
                    with xr.open_dataset(nc_file) as ds:
                        _ = ds["precipitation"].values[0]  # 仅检查首值

                    # 进度条颜色标记
                    pbar.set_postfix(status="✅")
                except Exception as e:
                    error_files.append(nc_file)
                    pbar.set_postfix(status="❌")
                    tqdm.write(f"ERROR: {nc_file} —— {str(e)[:50]}...")
                finally:
                    pbar.update(1)

        # 结果汇总
        if error_files:
            error_rate = len(error_files) / len(all_nc_files) * 100
            err_msg = (
                f"数据检查未通过！错误率 {error_rate:.2f}%\n损坏文件示例:\n"
                + "\n".join(error_files[:3])
            )
            if len(error_files) > 3:
                err_msg += f"\n...等共 {len(error_files)} 个错误文件"
            raise ValueError(err_msg)
        print(f"✅ 全部 {len(all_nc_files)} 个文件检查通过")

    def read_ts_xrdataset(
        self,
        time_range: list,
        period: str = "Past",
        time_resolution: str = "Daily",
        bbox: list = None,
        **kwargs,
    ) -> xr.Dataset:
        """优化后的MSWEP数据读取方法"""
        # 参数验证和时间解析
        if period not in ["NRT", "Past"]:
            raise ValueError("period必须为'NRT'或'Past'")
        if time_resolution not in ["Daily", "3hourly", "Monthly"]:
            raise ValueError("时间分辨率必须为'Daily', '3hourly'或'Monthly'")

        start_dt = pd.to_datetime(time_range[0]).tz_localize(None)  # 转换为无时区时间
        end_dt = pd.to_datetime(time_range[1]).tz_localize(None)

        # 获取数据目录
        data_dir = self.data_source_description["PERIOD_DATA"][period][time_resolution]

        # 生成需要匹配的文件名模式
        file_patterns = self._generate_file_patterns(start_dt, end_dt, time_resolution)

        # 收集匹配的文件
        matched_files = []
        for pattern in file_patterns:
            full_pattern = str(Path(data_dir) / pattern)
            matched_files.extend(glob.glob(full_pattern))

        matched_files = sorted(list(set(matched_files)))  # 去重并排序

        if not matched_files:
            raise FileNotFoundError(
                f"在{data_dir}中未找到{start_dt}到{end_dt}期间的数据文件"
            )

        # 智能合并数据集
        ds = self._smart_merge(matched_files, time_resolution)

        # 精确时间筛选
        ds = ds.sel(time=slice(start_dt, end_dt))

        # 空间裁剪优化
        if bbox:
            ds = self._optimize_spatial_slice(ds, bbox)

        return ds

    def _generate_file_patterns(self, start_dt, end_dt, time_res):
        """根据时间范围和分辨率生成文件名匹配模式"""
        patterns = []
        delta = pd.Timedelta(days=1)

        # 生成日期序列
        date_seq = pd.date_range(start_dt.floor("D"), end_dt.ceil("D"), freq=delta)

        for dt in date_seq:
            year = dt.year
            doy = dt.dayofyear  # 年序日

            if time_res == "Daily":
                # 日数据：YYYYddd.nc 如2000001.nc
                patterns.append(f"{year}{doy:03d}.nc")
            elif time_res == "3hourly":
                # 3小时数据：YYYYddd.HH.nc 如2000001.03.nc
                for hour in [0, 3, 6, 9, 12, 15, 18, 21]:
                    patterns.append(f"{year}{doy:03d}.{hour:02d}.nc")
            elif time_res == "Monthly":
                # 月数据：YYYYMM.nc 如200001.nc
                patterns.append(f"{year}{dt.month:02d}.nc")

        return list(set(patterns))  # 去重

    def _smart_merge(self, file_list, time_res):
        """智能合并策略"""
        # 构建时间索引映射
        time_index = []
        for f in file_list:
            fname = Path(f).stem
            if time_res == "Daily":
                time_str = fname[:7]
                year = int(time_str[:4])
                doy = int(time_str[4:])
                dt = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(
                    days=doy - 1
                )
            elif time_res == "3hourly":
                time_str, hour = fname.split(".")
                year = int(time_str[:4])
                doy = int(time_str[4:7])
                base_dt = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(
                    days=doy - 1
                )
                dt = base_dt + pd.Timedelta(hours=int(hour))
            elif time_res == "Monthly":
                year = int(fname[:4])
                month = int(fname[4:6])
                dt = pd.Timestamp(year=year, month=month, day=1)

            time_index.append(dt)

        # 创建虚拟时间索引
        # time_coord = xr.Coordinates(
        #     "time", pd.DatetimeIndex(time_index), {"long_name": "time"}
        # )

        # 并行读取优化
        ds = xr.open_mfdataset(
            file_list,
            combine="nested",
            concat_dim="time",
            parallel=True,
            chunks={"time": 100},
            decode_times=False,
        )

        # 重建正确的时间坐标
        ds["time"] = time_index
        return ds

    def _optimize_spatial_slice(self, ds, bbox):
        """优化空间裁剪"""
        lon_min, lat_min, lon_max, lat_max = bbox
        # 确保经度在[-180, 180]范围
        ds = ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180))
        # 快速空间索引
        return ds.sel(
            lon=slice(np.floor(lon_min), np.ceil(lon_max)),
            lat=slice(np.floor(lat_max), np.ceil(lat_min)),  # 纬度降序
        ).interp(
            lon=np.arange(np.floor(lon_min), np.ceil(lon_max) + 0.1, 0.1),
            lat=np.arange(np.floor(lat_min), np.ceil(lat_max) + 0.1, 0.1)[::-1],
        )


if __name__ == "__main__":
    # 使用示例
    data_dir = r"D:\data\MSWEP_V280"
    mswep = Mswep(data_dir)
    # try:
    #     mswep.check_data_source()
    # except ValueError as e:
    #     print(f"数据完整性检查未通过:\n{e}")
    # 读取2010年1月的数据
    ds = mswep.read_ts_xrdataset(
        time_range=["2010-01-01", "2010-01-31"],
        period="Past",
        time_resolution="Daily",
        bbox=[100, 20, 120, 40],  # 示例空间范围
    )
    print(ds)
