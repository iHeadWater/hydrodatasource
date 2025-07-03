import os
from pathlib import Path
import sys


sys.path.append(os.path.dirname(Path(os.path.abspath(__file__)).parent))
from const4scripts import DATASET_DIR
from hydrodatasource.reader.rainfall_reader import RainfallReader

rainfall_data_dir = os.path.join(DATASET_DIR, "basin_songliao_pp_station_cleaned")
rainfall_reader = RainfallReader(rainfall_data_dir)
a_station_rainfall = rainfall_reader.read_basin_rainfall(basin_ids=["21401550"])
print(a_station_rainfall)
