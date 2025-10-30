"""
Examples demonstrating the Runtime Data Loading system.

This script shows how to use the new atomic, flexible data loading
interface for real-time hydrological model execution.

Author: Wenyu Ouyang
Date: 2025-01-22
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# Add hydromodel to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hydrodatasource.runtime import RuntimeDataLoader, load_runtime_data


def create_sample_data():
    """Create sample hydrological data for examples."""
    print("Creating sample data...")

    # Create date range
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 1, 31)
    dates = pd.date_range(start_date, end_date, freq="D")

    # Create sample basins
    basin_ids = ["basin_001", "basin_002", "basin_003"]

    # Generate synthetic data
    np.random.seed(42)  # For reproducible results
    all_data = []

    for basin_id in basin_ids:
        for date in dates:
            # Generate realistic hydrological data
            prcp = max(0, np.random.normal(2.5, 3.0))  # Precipitation (mm/day)
            pet = max(0, np.random.normal(3.0, 1.0))  # Potential ET (mm/day)
            temp = np.random.normal(15.0, 5.0)  # Temperature (°C)

            # Generate streamflow based on precipitation (simplified)
            base_flow = 10.0
            flow_response = prcp * 0.3 + np.random.normal(0, 1)
            streamflow = max(0, base_flow + flow_response)

            all_data.append(
                {
                    "time": date,
                    "basin": basin_id,
                    "prcp": round(prcp, 2),
                    "PET": round(pet, 2),
                    "streamflow": round(streamflow, 2),
                    "temperature": round(temp, 1),
                }
            )

    df = pd.DataFrame(all_data)

    # Create sample CSV file
    csv_path = Path("sample_hydro_data.csv")
    df.to_csv(csv_path, index=False)
    print(f"Created sample CSV: {csv_path}")

    # Create sample data for memory loading
    sample_dict = {}
    for basin_id in basin_ids[:2]:  # Just first 2 basins
        basin_data = df[df["basin"] == basin_id].copy()
        basin_data = basin_data.drop("basin", axis=1).set_index("time")
        sample_dict[basin_id] = basin_data

    return csv_path, sample_dict, df


def example_1_basic_csv_loading():
    """Example 1: Basic CSV file loading."""
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Basic CSV Loading")
    print("=" * 60)

    loader = RuntimeDataLoader()

    # Load data from CSV
    print("Loading precipitation and streamflow data...")
    data = loader.load_variables(
        variables=["prcp", "streamflow"],
        basin_ids=["basin_001", "basin_002"],
        time_range=("2024-01-01", "2024-01-15"),
        source_type="csv",
        source_config={"file_path": "sample_hydro_data.csv"},
    )

    print(f"Loaded data shape: {data.shape}")
    print(f"Data columns: {data.columns.tolist()}")
    print("\nFirst few rows:")
    print(data.head())

    print("\nData summary:")
    print(data.describe())


def example_2_memory_loading():
    """Example 2: Memory-based data loading."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Memory Data Loading")
    print("=" * 60)

    # Create sample in-memory data
    _, sample_dict, _ = create_sample_data()

    loader = RuntimeDataLoader()

    # Load from memory (dictionary format)
    print("Loading data from memory (dictionary)...")
    data = loader.quick_load_memory(
        data=sample_dict,
        variables=["prcp", "PET", "streamflow"],
        basin_ids=["basin_001"],
        time_range=("2024-01-10", "2024-01-20"),
    )

    print(f"Loaded data shape: {data.shape}")
    print("\nData preview:")
    print(data.head())

    # Load from DataFrame
    print("\nLoading data from memory (DataFrame)...")
    df_sample = pd.concat(
        [
            sample_dict["basin_001"].reset_index().assign(basin="basin_001"),
            sample_dict["basin_002"].reset_index().assign(basin="basin_002"),
        ]
    )

    data2 = loader.quick_load_memory(
        data=df_sample,
        variables=["temperature"],
        basin_ids=["basin_001", "basin_002"],
        time_range=("2024-01-01", "2024-01-05"),
    )

    print(f"DataFrame data shape: {data2.shape}")
    print(data2.head())


def example_3_array_output_format():
    """Example 3: Array output format for model compatibility."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Array Output Format")
    print("=" * 60)

    # Load data in array format (compatible with existing models)
    print("Loading data in array format for model input...")
    p_and_e, qobs = load_runtime_data(
        variables=["prcp", "PET", "streamflow"],
        basin_ids=["basin_001", "basin_002"],
        time_range=("2024-01-01", "2024-01-10"),
        source_type="csv",
        source_config={"file_path": "sample_hydro_data.csv"},
        return_format="arrays",
    )

    print(f"p_and_e shape: {p_and_e.shape}")  # [time, basin, features=2]
    print(f"qobs shape: {qobs.shape}")  # [time, basin, features=1]

    print("\nFirst time step data:")
    print("p_and_e[0] (Precipitation and PET):")
    print(p_and_e[0])
    print("qobs[0] (Streamflow):")
    print(qobs[0])


def example_4_validation_and_info():
    """Example 4: Data validation and source information."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Validation and Information")
    print("=" * 60)

    loader = RuntimeDataLoader()

    # Validate a data request
    print("Validating data request...")
    validation = loader.validate_request(
        variables=["prcp", "streamflow", "invalid_var"],
        basin_ids=["basin_001", "invalid_basin"],
        time_range=("2024-01-01", "2025-01-01"),  # Future date
        source_type="csv",
        source_config={"file_path": "sample_hydro_data.csv"},
    )

    print("Validation results:")
    for key, value in validation.items():
        print(f"  {key}: {value}")

    # Get source information
    print("\nGetting source information...")
    source_info = loader.get_source_info(
        source_type="csv", source_config={"file_path": "sample_hydro_data.csv"}
    )

    print("Source information:")
    for key, value in source_info.items():
        if isinstance(value, (list, dict)) and len(str(value)) > 100:
            print(f"  {key}: {type(value).__name__} (size: {len(value)})")
        else:
            print(f"  {key}: {value}")


def example_5_streaming_simulation():
    """Example 5: Simulated real-time streaming data."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Streaming Data Simulation")
    print("=" * 60)

    from hydrodatasource.runtime.data_sources import StreamDataSource

    # Create stream data source
    stream_config = {
        "stream_type": "callback",
        "buffer_size": 1000,
        "buffer_duration": 3600,  # 1 hour buffer
        "variables": ["prcp", "streamflow"],
        "basin_ids": ["basin_001"],
    }

    stream_source = StreamDataSource(stream_config)

    # Simulate real-time data points
    print("Simulating real-time data stream...")
    base_time = datetime.now()

    for i in range(10):
        data_point = {
            "time": base_time + timedelta(minutes=i * 15),  # 15-minute intervals
            "basin": "basin_001",
            "prcp": max(0, np.random.normal(1.0, 2.0)),
            "streamflow": max(0, np.random.normal(15.0, 5.0)),
        }
        stream_source.push_data(data_point)
        print(
            f"  Pushed data point {i+1}: prcp={data_point['prcp']:.2f}, flow={data_point['streamflow']:.2f}"
        )

    # Load recent data from stream
    print("\nLoading data from stream buffer...")
    end_time = base_time + timedelta(hours=1)
    stream_data = stream_source.load_variables(
        variables=["prcp", "streamflow"],
        basin_ids=["basin_001"],
        time_range=(base_time, end_time),
    )

    print(f"Stream data shape: {stream_data.shape}")
    print(stream_data)

    # Get buffer information
    buffer_info = stream_source.get_buffer_info()
    print("\nBuffer information:")
    for key, value in buffer_info.items():
        print(f"  {key}: {value}")


def example_6_performance_comparison():
    """Example 6: Performance comparison with traditional loading."""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Performance Comparison")
    print("=" * 60)

    import time

    # Test runtime loader performance
    loader = RuntimeDataLoader()

    print("Testing RuntimeDataLoader performance...")
    start_time = time.time()

    for i in range(5):
        data = loader.load_variables(
            variables=["prcp", "streamflow"],
            basin_ids=["basin_001"],
            time_range=("2024-01-01", "2024-01-31"),
            source_type="csv",
            source_config={"file_path": "sample_hydro_data.csv"},
        )

    runtime_time = time.time() - start_time
    print(f"RuntimeDataLoader: 5 loads in {runtime_time:.3f} seconds")

    # Test traditional pandas loading
    print("Testing traditional pandas loading...")
    start_time = time.time()

    for i in range(5):
        df = pd.read_csv("sample_hydro_data.csv")
        df["time"] = pd.to_datetime(df["time"])
        df = df[
            (df["time"] >= "2024-01-01")
            & (df["time"] <= "2024-01-31")
            & (df["basin"] == "basin_001")
        ]
        df = df.set_index(["time", "basin"])
        data = df[["prcp", "streamflow"]]

    pandas_time = time.time() - start_time
    print(f"Traditional pandas: 5 loads in {pandas_time:.3f} seconds")

    print(f"\nSpeedup factor: {pandas_time/runtime_time:.2f}x")

    # Get cache statistics
    cache_stats = loader._active_sources
    print(f"Active data sources cached: {len(cache_stats)}")


def cleanup_sample_files():
    """Clean up sample files created for examples."""
    files_to_remove = ["sample_hydro_data.csv"]

    for filename in files_to_remove:
        file_path = Path(filename)
        if file_path.exists():
            file_path.unlink()
            print(f"Cleaned up: {filename}")


def main():
    """Run all examples."""
    print("Runtime Data Loading Examples")
    print("============================")
    print("\nThis demonstrates the new atomic, flexible data loading")
    print("interface for real-time hydrological model execution.")

    try:
        # Create sample data
        create_sample_data()

        # Run examples
        example_1_basic_csv_loading()
        example_2_memory_loading()
        example_3_array_output_format()
        example_4_validation_and_info()
        example_5_streaming_simulation()
        example_6_performance_comparison()

        print("\n" + "=" * 60)
        print("KEY ADVANTAGES OF RUNTIME DATA LOADING:")
        print("=" * 60)
        print("1. ATOMIC INTERFACE: Load only what you need, when you need it")
        print("2. MULTI-SOURCE: Files, databases, streams, memory - unified interface")
        print("3. FAST RESPONSE: Optimized for real-time performance with caching")
        print("4. SIMPLE USAGE: One-line data loading, minimal configuration")
        print(
            "5. FLEXIBLE OUTPUT: DataFrame or arrays, compatible with existing models"
        )
        print("\nCompare with traditional UnifiedDataLoader:")
        print("- Traditional: Complex config-driven, batch-oriented")
        print("- Runtime: Simple parameters, real-time optimized")

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # Cleanup
        cleanup_sample_files()

    print("\n" + "=" * 60)
    print("EXAMPLES COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
