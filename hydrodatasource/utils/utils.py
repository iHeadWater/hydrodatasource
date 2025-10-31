import re
import numpy as np
import geopandas as gpd
from netCDF4 import Dataset, date2num, num2date
import time
from datetime import datetime, timedelta
import pint
import xarray as xr
import contextlib
import tempfile
from ..configs.config import FS

from hydroutils.hydro_time import calculate_utc_offset


def creatspinc(value, data_vars, lats, lons, starttime, filename, resolution):
    gridspi = Dataset(filename, "w", format="NETCDF4")

    # dimensions
    gridspi.createDimension("time", value[0].shape[0])
    gridspi.createDimension("lat", value[0].shape[2])  # len(lat)
    gridspi.createDimension("lon", value[0].shape[1])

    # Create coordinate variables for dimensions
    times = gridspi.createVariable("time", np.float64, ("time",))
    latitudes = gridspi.createVariable("lat", np.float32, ("lat",))
    longitudes = gridspi.createVariable("lon", np.float32, ("lon",))

    # Create the actual variable
    for var, attr in data_vars.items():
        gridspi.createVariable(
            var,
            np.float32,
            (
                "time",
                "lon",
                "lat",
            ),
        )

    # Global Attributes
    gridspi.description = "var"
    gridspi.history = f"Created {time.ctime(time.time())}"
    gridspi.source = "netCDF4 python module tutorial"

    # Variable Attributes
    latitudes.units = "degree_north"
    longitudes.units = "degree_east"
    times.units = "days since 1970-01-01 00:00:00"
    times.calendar = "gregorian"

    # data
    latitudes[:] = lats
    longitudes[:] = lons

    # Fill in times
    dates = []
    if resolution == "daily":
        for n in range(value[0].shape[0]):
            dates.append(starttime + n)
        times[:] = dates[:]

    elif resolution == "6-hourly":
        # for n in range(value[0].shape[0]):
        #     dates.append(starttime + (n+1) * np.timedelta64(6, 'h'))

        for n in range(value[0].shape[0]):
            dates.append(starttime + (n + 1) * timedelta(hours=6))

        times[:] = date2num(dates, units=times.units, calendar=times.calendar)
        # print 'time values (in units %s): ' % times.units +'\n', times[:]
        dates = num2date(times[:], units=times.units, calendar=times.calendar)

    # Fill in values
    i = 0
    for var, attr in data_vars.items():
        gridspi.variables[var].long_name = attr["long_name"]
        gridspi.variables[var].units = attr["units"]
        gridspi.variables[var][:] = value[i][:]
        i = i + 1

    gridspi.close()


def regen_box(bbox, resolution, offset):
    lx = bbox[0]
    rx = bbox[2]
    LLON = np.round(
        int(lx)
        + resolution * int((lx - int(lx)) / resolution + 0.5)
        + offset
        * (int(lx * 10) / 10 + offset - lx)
        / abs(int(lx * 10) // 10 + offset - lx + 0.0000001),
        3,
    )
    RLON = np.round(
        int(rx)
        + resolution * int((rx - int(rx)) / resolution + 0.5)
        - offset
        * (int(rx * 10) / 10 + offset - rx)
        / abs(int(rx * 10) // 10 + offset - rx + 0.0000001),
        3,
    )

    by = bbox[1]
    ty = bbox[3]
    BLAT = np.round(
        int(by)
        + resolution * int((by - int(by)) / resolution + 0.5)
        + offset
        * (int(by * 10) / 10 + offset - by)
        / abs(int(by * 10) // 10 + offset - by + 0.0000001),
        3,
    )
    TLAT = np.round(
        int(ty)
        + resolution * int((ty - int(ty)) / resolution + 0.5)
        - offset
        * (int(ty * 10) / 10 + offset - ty)
        / abs(int(ty * 10) // 10 + offset - ty + 0.0000001),
        3,
    )

    # print(LLON,BLAT,RLON,TLAT)
    return [LLON, BLAT, RLON, TLAT]


def validate(date_text, formatter, error):
    try:
        return datetime.strptime(date_text, formatter)
    except ValueError as e:
        raise ValueError(error) from e


def cf2datetime(ds):
    ds = ds.copy()
    time_tmp1 = ds.indexes["time"]
    attrs = ds.coords["time"].attrs
    time_tmp2 = []
    for i in range(time_tmp1.shape[0]):
        tmp = time_tmp1[i]
        a = str(tmp.year).zfill(4)
        b = str(tmp.month).zfill(2)
        c = str(tmp.day).zfill(2)
        d = str(tmp.hour).zfill(2)
        e = str(tmp.minute).zfill(2)
        f = str(tmp.second).zfill(2)
        time_tmp2.append(np.datetime64(f"{a}-{b}-{c} {d}:{e}:{f}.00000000"))
    ds = ds.assign_coords(time=time_tmp2)
    ds.coords["time"].attrs = attrs

    return ds


def generate_time_intervals(start_date, end_date):
    # Initialize an empty list to store the intervals
    intervals = []

    # Loop over days
    while start_date <= end_date:
        # Loop over the four time intervals in a day
        intervals.extend(
            [start_date.strftime("%Y-%m-%d"), hour] for hour in ["00", "06", "12", "18"]
        )
        # Move to the next day
        start_date += timedelta(days=1)

    return intervals


def _convert_target_unit(target_unit):
    """Convert user-friendly unit to standard unit for internal calculations."""
    if match := re.match(r"mm/(\d+)(h|d)", target_unit):
        num, unit = match.groups()
        return int(num), unit
    return None, None


def _process_custom_unit(streamflow_data, custom_unit):
    """Process streamflow data with custom unit format like mm/3h."""
    custom_unit_pattern = re.compile(r"mm/(\d+)(h|d)")
    if custom_match := custom_unit_pattern.match(custom_unit):
        num, unit = custom_match.groups()
        if unit == "h":
            standard_unit = "mm/h"
            conversion_factor = int(num)
        elif unit == "d":
            standard_unit = "mm/d"
            conversion_factor = int(num)
        else:
            raise ValueError(f"Unsupported unit: {unit}")

        # Convert custom unit to standard unit
        if isinstance(streamflow_data, xr.Dataset):
            # For xarray, modify the data and attributes
            result = streamflow_data / conversion_factor
            result[list(result.keys())[0]].attrs["units"] = standard_unit
            return result
        else:
            # For numpy/pandas, just return the converted values
            return streamflow_data / conversion_factor, standard_unit
    else:
        # If it's not a custom unit format, return as is
        if isinstance(streamflow_data, xr.Dataset):
            result = streamflow_data.copy()
            result[list(result.keys())[0]].attrs["units"] = custom_unit
            return result
        else:
            return streamflow_data, custom_unit


def _get_unit_conversion_info(unit_str):
    """Get conversion information for a unit string.

    Returns:
        tuple: (standard_unit, conversion_factor) where conversion_factor
               is used to convert from standard unit to custom unit.
    """
    if not (match := re.match(r"mm/(\d+)(h|d)", unit_str)):
        # For standard units, no conversion needed
        return unit_str, 1
    num, unit = match.groups()
    if unit == "h":
        return "mm/h", int(num)
    elif unit == "d":
        return "mm/d", int(num)
    else:
        raise ValueError(f"Unsupported unit: {unit}")


def _get_actual_source_unit(streamflow_data, source_unit=None):
    """Determine the actual source unit from streamflow data.

    Parameters
    ----------
    streamflow_data : xarray.Dataset, pint.Quantity, numpy.ndarray,
                      pandas.DataFrame/Series
        The streamflow data to extract units from
    source_unit : str, optional
        Explicitly provided source unit that overrides data units

    Returns
    -------
    str or None
        The actual source unit string, or None if no unit information found
    """
    if source_unit is not None:
        return source_unit

    if isinstance(streamflow_data, xr.Dataset):
        streamflow_key = list(streamflow_data.keys())[0]
        # First check attrs for units
        if "units" in streamflow_data[streamflow_key].attrs:
            return streamflow_data[streamflow_key].attrs["units"]
        # Then check if it has pint units
        try:
            return str(streamflow_data[streamflow_key].pint.units)
        except (AttributeError, ValueError):
            return None
    elif isinstance(streamflow_data, pint.Quantity):
        return str(streamflow_data.units)
    else:
        # numpy array or pandas without units
        return None


def _normalize_unit(unit_str):
    """Normalize unit string for comparison (handle m3/s vs m^3/s and pint format)."""
    if not unit_str:
        return unit_str

    # Handle pint verbose format
    normalized = unit_str.replace("meter ** 3 / second", "m^3/s")
    normalized = normalized.replace("meter**3/second", "m^3/s")
    normalized = normalized.replace("cubic_meter / second", "m^3/s")
    normalized = normalized.replace("cubic_meter/second", "m^3/s")

    # Handle short format variations
    normalized = normalized.replace("m3/s", "m^3/s")
    normalized = normalized.replace("ft3/s", "ft^3/s")
    normalized = normalized.replace("ft**3/s", "ft^3/s")
    normalized = normalized.replace("cubic_foot / second", "ft^3/s")
    normalized = normalized.replace("cubic_foot/second", "ft^3/s")

    # Handle pint format for depth units
    normalized = normalized.replace("millimeter / day", "mm/d")
    normalized = normalized.replace("millimeter/day", "mm/d")
    normalized = normalized.replace("millimeter / hour", "mm/h")
    normalized = normalized.replace("millimeter/hour", "mm/h")

    return normalized


def _is_inverse_conversion(source_unit, target_unit):
    """Determine if this should be an inverse conversion based on units.

    Returns True if converting from depth units (mm/time) to volume units
    (m^3/s).
    Returns False if converting from volume units to depth units.
    """
    source_norm = _normalize_unit(source_unit) if source_unit else ""
    target_norm = _normalize_unit(target_unit)

    # Define unit patterns
    depth_pattern = re.compile(r"mm/(?:\d+)?[hd]?(?:ay|our)?$")
    volume_pattern = re.compile(r"(?:m\^?3|ft\^?3)/s$")

    source_is_depth = bool(depth_pattern.match(source_norm))
    source_is_volume = bool(volume_pattern.match(source_norm))
    target_is_depth = bool(depth_pattern.match(target_norm))
    target_is_volume = bool(volume_pattern.match(target_norm))

    if source_is_depth and target_is_volume:
        return True
    elif source_is_volume and target_is_depth:
        return False
    else:
        # If we can't determine from units, return None to indicate ambiguity
        return None


def _validate_inverse_consistency(source_unit, target_unit, inverse_param):
    """Validate that the inverse parameter is consistent with the units.

    Parameters
    ----------
    source_unit : str
        Source unit string
    target_unit : str
        Target unit string
    inverse_param : bool
        The inverse parameter provided by user

    Raises
    ------
    ValueError
        If inverse parameter is inconsistent with unit conversion direction
    """
    expected_inverse = _is_inverse_conversion(source_unit, target_unit)

    if expected_inverse is not None and expected_inverse != inverse_param:
        direction = "depth->volume" if expected_inverse else "volume->depth"
        raise ValueError(
            f"Inverse parameter ({inverse_param}) is inconsistent with unit "
            f"conversion direction. Converting from '{source_unit}' to "
            f"'{target_unit}' suggests {direction} conversion "
            f"(inverse={expected_inverse})."
        )


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
    try:
        # the minio folder url doesn't have to start with s3://, but we agree that it must
        # start with s3:// to distinguish between local and Minio folder directories.
        files = FS.ls(minio_folder_url)
        return [file.split("/")[-1] for file in files if not file.endswith("/")]
    except Exception as e:
        print(f"Error accessing {minio_folder_url}: {e}")
        return []


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
    try:
        if not FS.exists(minio_url):
            raise FileNotFoundError(f"No file or folder found in {minio_url}")
        if minio_url.endswith("/"):
            # If the path ends with '/', treat it as a directory
            return True
        # Try to list objects under this path
        objects = FS.ls(minio_url)
        test_object = "s3://" + objects[0]
        return len(objects) != 1 or test_object != minio_url
    except Exception as e:
        raise NotImplementedError(f"Error accessing {minio_url}: {e}") from e


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
