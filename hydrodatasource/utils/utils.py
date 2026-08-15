import contextlib
import tempfile
from datetime import datetime

import geopandas as gpd

from ..configs.config import FS
from hydroutils.hydro_time import calculate_utc_offset


def validate(date_text, formatter, error):
    try:
        return datetime.strptime(date_text, formatter)
    except ValueError as e:
        raise ValueError(error) from e


def streamflow_unit_conv(
    streamflow,
    area,
    target_unit="mm/d",
    inverse=False,
    source_unit=None,
    area_unit="km^2",
):
    """Convert the unit of streamflow data from m^3/s or ft^3/s to mm/xx(time) for a basin or inverse.

    This function is now a wrapper around the implementation in hydroutils for backward compatibility.

    .. deprecated::
        This function is deprecated and will be removed in the next version.
        Please use `hydroutils.hydro_units.streamflow_unit_conv` directly instead.

    Parameters
    ----------
    streamflow: xarray.Dataset, numpy.ndarray, pandas.DataFrame/Series, or pint.Quantity
        Streamflow data of each basin.
    area: xarray.Dataset, pint.Quantity, numpy.ndarray, pandas.DataFrame/Series
        Area of each basin. Can be with or without units.
    target_unit: str
        The unit to convert to.
    inverse: bool
        If True, convert the unit to m^3/s.
        If False, convert the unit to mm/day or mm/h.
    source_unit: str, optional
        The source unit of streamflow data. Use this when streamflow doesn't have
        unit information or when the unit is a custom format like 'mm/3h' that
        pint cannot recognize directly. If None, the function will try to get
        unit information from streamflow data attributes.
    area_unit: str, optional
        The unit of area data when area is provided without units (e.g., numpy array).
        Default is "km^2". Only used when area doesn't have unit information.

    Returns
    -------
    Converted data in the same type as the input streamflow.
    For numpy arrays, returns numpy array directly.
    """
    import warnings

    warnings.warn(
        "streamflow_unit_conv is deprecated and will be removed in the next version. "
        "Please use hydroutils.hydro_units.streamflow_unit_conv directly instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    # Import the new implementation from hydroutils
    try:
        from hydroutils.hydro_units import (
            streamflow_unit_conv as hydro_streamflow_unit_conv,
            _detect_data_unit,
            _validate_inverse_consistency,
        )

        # Detect source unit if not provided
        if source_unit is None:
            source_unit = _detect_data_unit(streamflow, source_unit)

        # Validate that the inverse parameter is consistent with unit conversion direction
        _validate_inverse_consistency(source_unit, target_unit, inverse)

        # Call the new hydroutils version with simplified interface
        return hydro_streamflow_unit_conv(
            data=streamflow,
            area=area,
            target_unit=target_unit,
            source_unit=source_unit,
            area_unit=area_unit,
        )
    except ImportError as e:
        # If hydroutils is not available, fall back to error message
        # This ensures backward compatibility during transition
        raise ImportError(
            f"hydroutils is not available. Please install hydroutils to use streamflow_unit_conv. "
            f"Original error: {e}"
        )


def minio_file_list(minio_folder_url):
    """
    Get all filenames in a specified directory on MinIO.

    Parameters
    ----------
    minio_folder_url : str
        the minio file url, must start with s3://

    Returns
    -------
    folder list
    """
    # Get the list of files in the directory
    # the minio folder url doesn't have to start with s3://, but we agree that it must
    # start with s3:// to distinguish between local and Minio folder directories.
    if FS is None:
        # S3 not configured in this environment; nothing to list
        return []
    files = FS.ls(minio_folder_url)
    return [file.split("/")[-1] for file in files if not file.endswith("/")]


def is_minio_folder(minio_url):
    """
    Check if a MinIO folder exists.

    Parameters
    ----------
    minio_url : str
        the minio file url, must start with s3://

    Returns
    -------
    bool
        True if the folder exists, False otherwise

    """
    if FS is None:
        # S3 not configured in this environment; nothing can be a MinIO folder
        return False
    if not FS.exists(minio_url):
        raise FileNotFoundError(f"No file or folder found in {minio_url}")
    if minio_url.endswith("/"):
        # If the path ends with '/', treat it as a directory
        return True
    # Try to list objects under this path
    objects = FS.ls(minio_url)
    # 存在但无子项 → 目录
    if not objects:
        return True
    # 归一化比较（s3fs ls 结果可能带或不带 s3:// 前缀）
    target = minio_url.rstrip("/").removeprefix("s3://")
    single = str(objects[0]).rstrip("/").removeprefix("s3://")
    # 只有自身一个子项 → 是文件；否则是目录
    return len(objects) != 1 or single != target


def calculate_basin_offsets(shp_file_path):
    """
    Calculate the UTC offset for each basin based on the outlet shapefile.

    Parameters:
        shp_file (str): The path to the basin outlet shapefile.

    Returns:
        dict: A dictionary where the keys are the BASIN_ID and the values are the corresponding UTC offsets.
    """
    # read shapefile
    if "s3://" in shp_file_path:
        # related list
        extensions = [".shp", ".shx", ".dbf", ".prj"]

        # create a temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            # download all related files to the temporary directory
            base_name = shp_file_path.rsplit(".", 1)[0]
            extensions = [".shp", ".shx", ".dbf", ".prj"]

            for ext in extensions:
                remote_file = f"{base_name}{ext}"
                local_file = f"{tmpdir}/shp_file{ext}"
                with contextlib.suppress(FileNotFoundError):
                    FS.get(remote_file, local_file)
            gdf = gpd.read_file(f"{tmpdir}/shp_file.shp")

    else:
        # If the file is not on S3 (MinIO), read it directly
        gdf = gpd.read_file(shp_file_path)

    # create an empty dictionary
    basin_offset_dict = {}

    for index, row in gdf.iterrows():
        outlet = row["geometry"]
        # TODO: Only for temp use.
        offset = calculate_utc_offset(
            outlet.y, outlet.x, datetime(2024, 8, 14, 0, 0, 0)
        )
        basin_id = row.get(
            "BASIN_ID", index
        )  # Use the index as the default value if "BASIN_ID" is not found
        basin_offset_dict[basin_id] = offset

    return basin_offset_dict


def cal_area_from_shp(shp):
    gdf_equal_area = shp.to_crs(epsg=6933)
    gdf_equal_area["shp_area"] = gdf_equal_area["geometry"].area / 10**6
    result_df = gdf_equal_area[["BASIN_ID", "shp_area"]]
    result_df.rename(columns={"BASIN_ID": "basin_id"}, inplace=True)
    result_df.sort_values("basin_id", inplace=True)
    return result_df
