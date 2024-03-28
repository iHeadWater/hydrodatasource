import os
import pandas as pd

from hydrodatasource.processor.data_checker import DataChecker


def test_check_folder_structure():
    """
    Test to check if the folder structure in the local data path matches the expected format.
    """
    # Initialize the DataChecker with the local data path from the settings
    data_checker = DataChecker()

    # Check the folder structure
    result = data_checker.check_folder_structure()

    # Assert that the result is correct
    assert result


def test_check_file_format(tmpdir):
    """
    Test to check if the file format is correct.
    """
    # Initialize the DataChecker with the local data path from the settings
    data_checker = DataChecker()

    # Define a sample file path and expected columns
    file_path = os.path.join(str(tmpdir), "file.csv")
    expected_columns = ["column1", "column2", "column3"]

    # Create a sample file with the expected columns
    sample_data = pd.DataFrame(columns=expected_columns)
    sample_data.to_csv(file_path, index=False)

    # Check the file format
    result = data_checker.check_file_format(file_path, expected_columns)

    # Assert that the result is True
    assert result

    # Clean up the sample file
    os.remove(file_path)


def test_check_files_in_folder(tmpdir):
    """
    Test to check the format of all CSV files in a folder.
    """
    # Initialize the DataChecker with the local data path from the settings
    data_checker = DataChecker()

    # Define a sample folder path and expected columns
    folder_path = str(tmpdir)
    expected_columns = ["column1", "column2", "column3"]

    # Create sample CSV files with the expected columns
    file1_path = os.path.join(folder_path, "file1.csv")
    file2_path = os.path.join(folder_path, "file2.csv")
    file3_path = os.path.join(folder_path, "file3.csv")

    sample_data = pd.DataFrame(columns=expected_columns)
    sample_data.to_csv(file1_path, index=False)
    sample_data.to_csv(file2_path, index=False)
    sample_data.to_csv(file3_path, index=False)

    # Check the file formats in the folder
    result = data_checker.check_files_in_folder(folder_path, expected_columns)

    # Assert that the result is True
    assert result

    # Clean up the sample files
    os.remove(file1_path)
    os.remove(file2_path)
    os.remove(file3_path)


def test_check_station_data_files(tmpdir):
    """
    Test to check the format of station data files.
    """
    # Initialize the DataChecker
    data_checker = DataChecker()

    # Define a sample folder path and expected columns
    folder_path = tmpdir.mkdir("stations-origin")
    expected_columns = {
        "pp_stations": ["column1", "column2", "column3"],
        "zz_stations": ["column4", "column5", "column6"],
        "zq_stations": ["column7", "column8", "column9"],
    }

    # Create directories and sample files for each station type
    for station_type in ["pp", "zz", "zq"]:
        # Create station directories
        station_dir = folder_path.mkdir(f"{station_type}_stations")
        basic_info_file = station_dir.join(f"{station_type}_stations.csv")
        time_series_file = station_dir.join("time_series.csv")

        # Create a sample file with the expected columns
        sample_data = pd.DataFrame(columns=expected_columns[f"{station_type}_stations"])
        sample_data.to_csv(str(basic_info_file), index=False)
        sample_data.to_csv(str(time_series_file), index=False)

    # Check the format of station data files
    result = data_checker.check_station_data_files()

    # Assert that the result is True
    assert result
