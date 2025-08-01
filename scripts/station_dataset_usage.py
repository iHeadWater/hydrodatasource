"""
StationHydroDataset Usage Example

This example demonstrates how to use the StationHydroDataset class
to read and cache station data from a hydro dataset.
"""

import os
import sys
import pandas as pd
import numpy as np

# Add the parent directory to sys.path to import hydrodatasource
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hydrodatasource.reader.data_source import StationHydroDataset


def main():
    # Initialize the dataset
    # Use actual data path for songliao dataset
    data_path = "/mnt/data/ClassC/songliaorrevent"

    # Create StationHydroDataset instance
    dataset = StationHydroDataset(
        data_path=data_path,
        time_unit=["1D", "3h"],  # Support both daily and 3-hourly data
        dataset_name="songliao_station_dataset",
        offset_to_utc=False,  # Convert Beijing time to UTC
    )

    print("=== StationHydroDataset Usage Example ===")

    # 1. Read basic station information
    print("\n1. Reading station information...")
    mapping_data, summary_data = dataset.read_station_info()
    print(f"Total basins: {len(summary_data)}")
    print(f"Total basin-station mappings: {len(mapping_data)}")

    # 2. Get all station IDs
    print("\n2. Getting all station IDs...")
    all_station_ids = dataset.read_station_object_ids()
    print(f"Total stations: {len(all_station_ids)}")
    print(f"First 5 stations: {all_station_ids[:5]}")

    # 3. Get stations for a specific basin
    print("\n3. Getting stations for a specific basin...")
    basin_id = "20810200"  # Replace with actual basin ID
    basin_stations = dataset.get_stations_by_basin(basin_id)
    print(f"Stations in basin {basin_id}: {basin_stations}")

    # 4. Read detailed station information for a basin
    print("\n4. Reading detailed station information...")
    try:
        station_details = dataset.read_basin_stations(basin_id)
        print(f"Station details columns: {station_details.columns.tolist()}")
        print(f"Number of stations: {len(station_details)}")
    except FileNotFoundError:
        print(f"No detailed station file found for basin {basin_id}")

    # 5. Read adjacency matrix for a basin
    print("\n5. Reading adjacency matrix...")
    try:
        adjacency_matrix = dataset.read_basin_adjacency(basin_id)
        print(f"Adjacency matrix shape: {adjacency_matrix.shape}")
    except FileNotFoundError:
        print(f"No adjacency matrix found for basin {basin_id}")

    # 6. Read station timeseries data
    print("\n6. Reading station timeseries data...")
    if len(basin_stations) > 0:
        # Select a few stations for demonstration
        sample_stations = (
            basin_stations[:3] if len(basin_stations) >= 3 else basin_stations
        )

        # Define time range and variables
        t_range = ["2020-01-01", "2020-01-31"]
        variables = ["streamflow", "water_level"]  # Replace with actual variable names

        try:
            station_data = dataset.read_station_timeseries(
                station_ids=sample_stations,
                t_range_list=t_range,
                relevant_cols=variables,
                time_units=["3h"],
            )

            for time_unit, data in station_data.items():
                print(f"Data shape for {time_unit}: {data.shape}")
                print(
                    f"  (stations: {data.shape[0]}, time: {data.shape[1]}, variables: {data.shape[2]})"
                )

        except Exception as e:
            print(f"Error reading station timeseries: {e}")

    # 7. Cache all station data
    print("\n7. Caching all station data...")
    try:
        dataset.cache_all_station_data(
            batchsize=50,  # Process 50 stations per batch
            time_units=["3h"],  # Cache 3-hourly data for testing
        )
        print("Station data cached successfully!")
    except Exception as e:
        print(f"Error caching station data: {e}")

    # 8. Read cached station data
    print("\n8. Reading cached station data...")
    try:
        cached_data = dataset.read_station_ts_xrdataset(
            station_id_lst=sample_stations,
            t_range=t_range,
            var_lst=variables,
            time_units=["3h"],
        )

        for time_unit, ds in cached_data.items():
            print(f"Cached dataset for {time_unit}:")
            print(f"  Variables: {list(ds.data_vars)}")
            print(f"  Stations: {len(ds.station)}")
            print(f"  Time points: {len(ds.time)}")

    except Exception as e:
        print(f"Error reading cached data: {e}")

    # 9. Read cached station info
    print("\n9. Reading cached station info...")
    try:
        mapping_ds, summary_ds = dataset.read_station_info_xrdataset()
        print(f"Mapping dataset variables: {list(mapping_ds.data_vars)}")
        print(f"Summary dataset variables: {list(summary_ds.data_vars)}")
    except Exception as e:
        print(f"Error reading cached station info: {e}")

    # 10. Test new adjacency matrix caching and reading functions
    print("\n10. Testing adjacency matrix functions...")
    try:
        # Cache adjacency matrices
        print("  Caching adjacency matrices...")
        dataset.cache_adjacency_xrdataset()
        print("  Adjacency matrices cached successfully!")
        
        # Read cached adjacency matrix
        print("  Reading cached adjacency matrix...")
        adjacency_ds = dataset.read_adjacency_xrdataset(basin_id)
        print(f"  Adjacency dataset variables: {list(adjacency_ds.data_vars)}")
        print(f"  Adjacency dataset coordinates: {list(adjacency_ds.coords)}")
        print(f"  Number of stations in adjacency: {len(adjacency_ds.station_from)}")
        
    except Exception as e:
        print(f"Error with adjacency matrix functions: {e}")

    print("\n=== Example completed ===")


def demonstrate_basin_analysis():
    """Demonstrate basin-level analysis using station data."""
    print("\n=== Basin Analysis Example ===")

    # This is a more advanced example showing how to analyze data by basin
    data_path = "/mnt/data/ClassC/songliaorrevent"

    dataset = StationHydroDataset(
        data_path=data_path,
        time_unit=["1D"],
        dataset_name="basin_analysis",
        version="v1.0",
    )

    # Get basin summary
    _, summary_data = dataset.read_station_info()

    # Analyze each basin
    for _, basin_row in summary_data.iterrows():
        basin_id = basin_row["basin_id"]
        print(f"\nAnalyzing basin: {basin_id}")

        # Get stations in this basin
        stations = dataset.get_stations_by_basin(basin_id)
        print(f"  Number of stations: {len(stations)}")

        if len(stations) > 0:
            try:
                # Read station details
                station_details = dataset.read_basin_stations(basin_id)
                print(f"  Station details columns: {station_details.columns.tolist()}")
                if 'data_type' in station_details.columns:
                    print(f"  Data types: {station_details['data_type'].value_counts().to_dict()}")

                # Read adjacency matrix
                adjacency = dataset.read_basin_adjacency(basin_id)
                print(f"  Network connectivity - adjacency shape: {adjacency.shape}")
                print(f"  Network columns: {adjacency.columns.tolist()}")

            except FileNotFoundError:
                print(f"  Detailed info not available for basin {basin_id}")


if __name__ == "__main__":
    main()
    demonstrate_basin_analysis()
