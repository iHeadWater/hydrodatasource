import json
import os

import dataretrieval.nwis as nwis
import geopandas as gpd
import numpy as np
import pandas as pd
import polars as pol
import requests
import tzfpy
import urllib3 as ur
import xarray as xr
from geopandas import points_from_xy
from pandas.errors import ParserError

from hydrodatasource.configs.config import FS
from hydrodatasource.reader.spliter_grid import read_streamflow_from_minio


def test_gen_camels_hourly_shp():
    camels_shp = "/ftproot/camels/camels_us/basin_set_full_res/HCDN_nhru_final_671.shp"
    camels_hourly_csvs = "/ftproot/camels_hourly/data/usgs_streamflow_csv/"
    csv_paths = os.listdir(camels_hourly_csvs)
    minio_csv_paths = FS.glob("s3://basins-origin/basin_shapefiles/**")[1:]
    minio_stcds = [csv.split("_")[-1].split(".")[0] for csv in minio_csv_paths]
    hourly_basin_ids = [path.split("-")[0] for path in csv_paths]
    camels_gpd = gpd.read_file(camels_shp)
    camels_gpd["hru_id"] = camels_gpd["hru_id"].astype("str")
    camels_gpd["hru_id"] = camels_gpd["hru_id"].apply(lambda x: x.zfill(8))
    camels_hourly_gpd_blank = camels_gpd[camels_gpd["hru_id"].isin(hourly_basin_ids)]
    camels_hourly_gpd = camels_hourly_gpd_blank[~camels_gpd["hru_id"].isin(minio_stcds)]
    camels_hourly_gpd.to_file("/home/jiaxuwu/camels_hourly_basins.shp")


def test_read_usa_streamflow():
    # basin_list = hdscc.FS.glob('s3://basins-origin/basin_shapefiles/**')
    basin_list = [
        "basin_USA_camels_01181000",
        "basin_USA_camels_01411300",
        "basin_USA_camels_01414500",
        "basin_USA_camels_02016000",
        "basin_USA_camels_02018000",
        "basin_USA_camels_02481510",
        "basin_USA_camels_03070500",
        "basin_USA_camels_08324000",
        "basin_USA_camels_11266500",
        "basin_USA_camels_11523200",
        "basin_USA_camels_12020000",
        "basin_USA_camels_12167000",
        "basin_USA_camels_14185000",
        "basin_USA_camels_14306500",
        "basin_CHN_songliao_21401550",
        "basin_USA_camels_14400000",
    ]
    for basin_id in basin_list:
        s3_basin_path = f"s3://basins-origin/basin_shapefiles/{basin_id}.zip"
        basin_gpd = gpd.read_file(FS.open(s3_basin_path))
        basin_tz = tzfpy.get_tz(
            basin_gpd.geometry[0].centroid.x, basin_gpd.geometry[0].centroid.y
        )
        q_array = read_streamflow_from_minio(
            times=[
                ["2014-12-31 17:00:00", "2019-12-31 23:00:00"],
                ["2020-01-01 00:00:00", "2023-12-31 23:00:00"],
            ],
            sta_id=basin_id.lstrip("basin_"),
        )
        if basin_tz == "America/Los_Angeles":
            q_array["TM"] = q_array["TM"] + np.timedelta64(7, "h")
        elif basin_tz == "America/Denver":
            q_array["TM"] = q_array["TM"] + np.timedelta64(6, "h")
        elif basin_tz == "America/Chicago":
            q_array["TM"] = q_array["TM"] + np.timedelta64(5, "h")
        elif basin_tz == "America/New_York":
            q_array["TM"] = q_array["TM"] + np.timedelta64(4, "h")
        q_array_2019 = q_array[q_array["TM"] < "2020-01-01 00:00:00"]
        q_array_2020 = q_array[q_array["TM"] >= "2020-01-01 00:00:00"]
        # convert degree^2 to km^2
        basin_area = gpd.read_file(FS.open(s3_basin_path)).geometry[0].area * 12345.6789
        # convert ft^3 to m^3
        q_array_2020["Q"] = q_array_2020["Q"] / 35.31
        # convert m^3 to mm/h
        q_array_2020["Q"] = q_array_2020["Q"] / basin_area * 3.6
        q_array_mm_h = pd.concat([q_array_2019, q_array_2020])
        q_ds = xr.Dataset.from_dataframe(q_array_mm_h.set_index("TM"))
        q_ds.to_netcdf(basin_id + "_streamflow.nc")


def test_download_from_usgs():
    """
    camels_hourly_csvs = '/ftproot/camels_hourly/data/usgs_streamflow_csv/'
    csv_paths = os.listdir(camels_hourly_csvs)
    """
    minio_csv_paths = FS.glob("s3://basins-origin/basin_shapefiles/**")[1:]
    # hourly_basin_ids = [path.split('-')[0] for path in csv_paths]
    minio_stcds = [csv.split("_")[-1].split(".")[0] for csv in minio_csv_paths]
    # empty_stcds = [stcd for stcd in hourly_basin_ids if stcd not in minio_stcds]
    # usgs_basins = gpd.read_file("/ftproot/usgs_camels_hourly_flowdb_1373/concat_flow_camels_1373.zip")
    # first_empty_stcds = usgs_basins['basin_id'].to_numpy()
    # usgs_gages = gpd.read_file("/ftproot/usgs_camels_hourly_flowdb_1373/concat_usa_usgs_all.shp")
    # empty_stcds = usgs_gages['GAGE_ID'].to_numpy()
    # empty_stcds = np.union1d(first_empty_stcds, second_empty_stcds)
    for i in range(len(minio_csv_paths)):
        site = minio_stcds[i]
        country = minio_csv_paths[i].split("_")[2]
        site_path = f"/ftproot/usgs_minio_basins/zq_USA_usgs_{site}.csv"
        if (not os.path.exists(site_path)) & (country == "USA"):
            try:
                site_df = nwis.get_record(sites=site, service="iv", start="2015-01-01")
            except requests.exceptions.ConnectionError:
                continue
            except requests.exceptions.JSONDecodeError:
                continue
            site_df.to_csv(site_path)


def test_gen_usgs_camels_gdf():
    basin_shp_gdf = gpd.read_file("iowa_all_locs/basins_shp.shp", engine="pyogrio")
    usa_basins = basin_shp_gdf["BASIN_ID"][
        basin_shp_gdf["country"] == "US"
    ].to_list() + [
        "08171300",
        "08164600",
        "06879650",
        "06746095",
        "05413500",
        "01022500",
        "02056900",
        "03574500",
        "03604000",
    ]
    try:
        site_df = nwis.get_record(sites=usa_basins, service="site")
    except requests.exceptions.ConnectionError:
        print("Failed!")
    camels_gdf = gpd.GeoDataFrame(
        geometry=points_from_xy(x=site_df["dec_long_va"], y=site_df["dec_lat_va"])
    ).set_crs(4326)
    camels_gdf["BASIN_ID"] = site_df["site_no"].astype("str")
    camels_gdf.to_file("iowa_all_locs/camels_us_basins.shp")


def test_gen_usgs_camels_gdf_2():
    usgs_points_gdf = gpd.read_file("iowa_all_locs/camels_us_basins.shp").to_crs(4326)
    usgs_points_gdf = usgs_points_gdf.rename(columns={"BASIN_ID": "ID"})
    iowa_stream_gdf = gpd.read_file("iowa_all_locs/iowa_stream_stations.shp").to_crs(
        4326
    )
    iowa_usgs_gdf = gpd.GeoDataFrame(
        pd.concat(
            [iowa_stream_gdf[["ID", "geometry"]], usgs_points_gdf[["ID", "geometry"]]]
        )
    )
    sl_dots_shp = gpd.read_file("sl_stcd_locs/100_sl_stcds.shp").rename(
        columns={"STCD": "ID"}
    )
    sl_usa_gdf = gpd.GeoDataFrame(pd.concat([iowa_usgs_gdf, sl_dots_shp]))
    sl_usa_gdf.to_file("sl_stcd_locs/iowa_usgs_sl_stations.shp")


def test_gen_hml_usgs_camels_gdf():
    usa_dots_shp = gpd.read_file("sl_stcd_locs/iowa_usgs_sl_stations.shp")
    hml_stcds = gpd.read_file("noaa_hml_inter_fdb_basins.shp")
    hml_stcds = hml_stcds.rename(columns={"LID": "ID"})
    hml_stcds["ID"] = hml_stcds["ID"].apply(lambda x: f"HML_{x}")
    hml_slice = hml_stcds[["ID", "geometry"]]
    total_gdf = gpd.GeoDataFrame(
        pd.concat([usa_dots_shp, hml_slice]).reset_index(drop=True)
    )
    total_gdf.to_file("flowdb_locs/fdb_iowa_usgs_hml_sl_stations.shp")


def test_dload_usgs_prcp_stations():
    import re

    with open("usgs_prcp_stations.txt", "r") as fp:
        stations_txt = fp.read()
    sta_list = re.findall("\d{8,}", stations_txt)
    # site_base_df = pd.DataFrame()
    for sta_id in sta_list:
        """
        try:
            site_basin_df = nwis.get_record(sites=sta_id, service='site')
        except ValueError:
            site_basin_df = pd.DataFrame()
        site_base_df = pd.concat([site_base_df, site_basin_df])
        """
        site_path = f"/ftproot/usgs_prcp_stations/pp_USA_usgs_{sta_id}.csv"
        if not os.path.exists(site_path):
            try:
                site_df = nwis.get_record(
                    sites=sta_id, service="iv", start="2000-01-01"
                )
            except requests.exceptions.ConnectionError:
                continue
            except requests.exceptions.JSONDecodeError:
                continue
            except requests.exceptions.ChunkedEncodingError:
                continue
            site_df.to_csv(site_path)
    # site_base_df.to_csv('usgs_prcp_stations_base.csv')


def test_gen_usgs_prcp_shp():
    csv_df = pd.read_csv("usgs_prcp_stations_base.csv")
    gpd_csv_df = gpd.GeoDataFrame(
        csv_df, geometry=gpd.points_from_xy(csv_df["dec_long_va"], csv_df["dec_lat_va"])
    )
    basin_shp_gdf = gpd.read_file("iowa_all_locs/basins_shp.shp", engine="pyogrio")
    camels_gpd_csv_df = gpd.sjoin(gpd_csv_df, basin_shp_gdf)
    camels_gpd_csv_df.to_file("usgs_prcp_shp/camels_usgs_prcp_stations.shp")


def test_gen_usgs_prcp_nc():
    # USGS变量意义：https://waterservices.usgs.gov/docs/site-service/site-service-details/
    usgs_camels_shp = gpd.read_file("usgs_prcp_shp/camels_usgs_prcp_stations.shp")
    total_site_df = pd.DataFrame()
    for i in range(len(usgs_camels_shp)):
        site_no = usgs_camels_shp["site_no"][i].astype(str).zfill(8)
        site_path = f"/ftproot/usgs_prcp_stations/pp_USA_usgs_{site_no}.csv"
        try:
            site_df = pd.read_csv(site_path, engine="c")
        except FileNotFoundError:
            site_df = nwis.get_record(sites=site_no, service="iv", start="2000-01-01")
            site_df.to_csv(site_path)
        if "datetime" not in site_df.columns:
            site_df = site_df.reset_index()
        site_df["datetime"] = (
            pd.to_datetime(site_df["datetime"])
            .dt.tz_convert("UTC")
            .dt.tz_localize(None)
        )
        total_site_df = pd.concat([total_site_df, site_df])
    total_site_df = total_site_df.set_index(["site_no", "datetime"])
    total_site_ds = xr.Dataset.from_dataframe(total_site_df)
    total_site_ds.to_netcdf("/ftproot/usgs_prcp_camels.nc")


def test_gen_iowa_paths():
    import hydrotopo.ig_path as htip

    usa_nodes_shp = gpd.read_file("iowa_all_locs/iowa_usgs_stations.shp")
    nw_shp = gpd.read_file(
        "/home/wangyang1/sl_sx_usa_shps/SL_USA_HydroRiver_single.shp", engine="pyogrio"
    )
    path = htip.find_edge_nodes(usa_nodes_shp, nw_shp, 100)
    print(path)


def test_dload_hml_metadata():
    if os.path.exists("gauges.json"):
        res = json.load(open("gauges.json", "r"))
    else:
        http = ur.PoolManager()
        r = http.request(
            "GET",
            "https://api.water.noaa.gov/nwps/v1/gauges?bbox.xmin=-130&bbox.ymin=25&bbox.xmax=-70&bbox.ymax=50&srid=EPSG_4326",
            timeout=1000,
        )
        res = json.loads(r.data)
        json.dump(res, open("gauges.json", "w"))
    lon_list = []
    lat_list = []
    name_list = []
    for i in range(len(res["gauges"])):
        lon_list.append(res["gauges"][i]["longitude"])
        lat_list.append(res["gauges"][i]["latitude"])
        name_list.append(res["gauges"][i]["lid"])
    gdf = gpd.GeoDataFrame(
        {"LID": name_list}, geometry=points_from_xy(x=lon_list, y=lat_list)
    )
    gdf.to_file("IOWA_NOAA_HML.shp")


def test_intersect_hml_data():
    ghcnh_locs = gpd.read_file("IOWA_NOAA_HML.shp").set_crs(4326)
    basin_shps = gpd.read_file("iowa_all_locs/basins_shp.shp")
    intersection = gpd.sjoin(ghcnh_locs, basin_shps)
    urpool = ur.PoolManager()
    error_list = []
    for name in intersection["LID"]:
        try:
            dload_data(urpool, name)
        except ParserError:
            error_list.append(name)
            continue
    other_names = [
        name
        for name in ghcnh_locs["LID"].to_list()
        if name not in intersection["LID"].to_list()
    ]
    for name in other_names:
        try:
            dload_data(urpool, name)
        except ParserError:
            error_list.append(name)
            continue
    np.save("error_hml.npy", np.array(error_list))


def dload_data(urpool, name):
    import io

    url_name = f"https://mesonet.agron.iastate.edu/cgi-bin/request/hml.py?station={name}&kind=obs&tz=UTC&year1=2012&month1=1&day1=1&year2=2024&month2=11&day2=25&fmt=csv"
    hml_path = f"/ftproot/hml_stations/hml_{name}.csv"
    if not os.path.exists(hml_path):
        req = urpool.request("GET", url_name, timeout=1200)
        df = pd.read_csv(io.BytesIO(req.data))
        df.to_csv(hml_path)


def test_calc_loss_rate_hml():
    rate_list = []
    name_list = []
    syear_list = []
    eyear_list = []
    hml_locs = gpd.read_file("IOWA_NOAA_HML.shp").set_crs(4326)
    basin_shps = gpd.read_file("flowdb_locs/new_basins_shp.shp")
    intersection = gpd.sjoin(hml_locs, basin_shps)
    # intersection.to_file('noaa_hml_inter_fdb_basins.shp')
    for name in intersection["LID"]:
        hml_path = f"/ftproot/hml_stations/hml_{name}.csv"
        if os.path.exists(hml_path):
            name_list.append(name)
            df = pol.read_csv(
                hml_path,
                schema_overrides={
                    "valid[UTC]": pol.Datetime,
                    "Flow[kcfs]": pol.Float32,
                },
                ignore_errors=True,
            )
            if "Flow[kcfs]" not in df.columns:
                rate_list.append(0)
                if len(df) != 0:
                    syear_list.append(df["valid[UTC]"].dt.year()[0])
                    eyear_list.append(df["valid[UTC]"].dt.year()[-1])
                else:
                    syear_list.append(9999)
                    eyear_list.append(0)
            else:
                nonnull_df = df.filter(~df["Flow[kcfs]"].is_nan())
                rate = len(nonnull_df) / len(df)
                rate_list.append(rate)
                syear_list.append(nonnull_df["valid[UTC]"].dt.year()[0])
                eyear_list.append(nonnull_df["valid[UTC]"].dt.year()[-1])
    rate_df = pol.DataFrame(
        {
            "LID": name_list,
            "rate": rate_list,
            "start_year": syear_list,
            "end_year": eyear_list,
        }
    )
    files_df = rate_df.filter(
        (rate_df["rate"] >= 0.2)
        & (rate_df["start_year"] <= 2023)
        & (rate_df["end_year"] >= 2023)
    )
    rate_df.write_csv("loss_rate_hml_fdb.csv")
    files_df.write_csv("ready_files_hml_fdb.csv")


def test_concat_hml_parquet():
    files_df = pol.read_csv("ready_files_hml_fdb.csv")
    total_df = pol.DataFrame()
    schema_error_list = []
    for name in files_df["LID"]:
        hml_path = f"/ftproot/hml_stations/hml_{name}.csv"
        hml_df = pol.read_csv(
            hml_path,
            schema_overrides={"valid[UTC]": pol.Datetime, "Flow[kcfs]": pol.Float32},
        )
        if "Stage[ft]" not in hml_df.columns:
            hml_df = hml_df.with_columns(pol.col("Pool[ft]").alias("Stage[ft]"))
        hml_df = hml_df.group_by_dynamic(
            index_column="valid[UTC]",
            every="1h",
            closed="both",
            include_boundaries=True,
        ).agg(
            pol.col("station").first(),
            pol.col("Stage[ft]").mean(),
            pol.col("Flow[kcfs]").mean(),
        )
        hml_df = hml_df[["station", "valid[UTC]", "Flow[kcfs]", "Stage[ft]"]]
        try:
            total_df = pol.concat([total_df, hml_df])
        except pol.exceptions.SchemaError:
            schema_error_list.append(name)
            continue
    print(schema_error_list)
    total_df.write_parquet("hml_flowdb_stations.parquet")
