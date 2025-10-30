"""
Author: Wenyu Ouyang
Date: 2025-08-12 17:36:17
LastEditTime: 2025-08-12 17:37:22
LastEditors: Wenyu Ouyang
Description: Simple test for runtime data loading to identify issues.
FilePath: \hydrodatasource\scripts\test_runtime_simple.py
Copyright (c) 2023-2026 Wenyu Ouyang. All rights reserved.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Add hydrodatasource to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hydrodatasource.runtime import RuntimeDataLoader


def test_basic_loading():
    """Test basic CSV loading."""
    print("=== Test 1: Basic CSV Loading ===")

    # Create simple test data
    data = {
        "time": pd.date_range("2024-01-01", periods=5),
        "basin": ["basin_001"] * 5,
        "prcp": [1.0, 2.0, 3.0, 4.0, 5.0],
        "streamflow": [10.0, 11.0, 12.0, 13.0, 14.0],
    }
    df = pd.DataFrame(data)
    df.to_csv("test_data.csv", index=False)

    # Test loading
    loader = RuntimeDataLoader()
    result = loader.load_variables(
        variables=["prcp", "streamflow"],
        basin_ids=["basin_001"],
        time_range=("2024-01-01", "2024-01-03"),
        source_type="csv",
        source_config={"file_path": "test_data.csv"},
    )

    print(f"Result shape: {result.shape}")
    print("Result:")
    print(result)
    print("SUCCESS: CSV loading works!")

    # Cleanup
    os.remove("test_data.csv")


def test_memory_loading():
    """Test memory data loading."""
    print("\n=== Test 2: Memory Loading ===")

    # Create test data
    df = pd.DataFrame(
        {
            "time": pd.date_range("2024-01-01", periods=3),
            "basin": ["basin_001"] * 3,
            "prcp": [1.0, 2.0, 3.0],
            "streamflow": [10.0, 11.0, 12.0],
        }
    )

    loader = RuntimeDataLoader()
    result = loader.quick_load_memory(
        data=df,
        variables=["prcp"],
        basin_ids=["basin_001"],
        time_range=("2024-01-01", "2024-01-02"),
    )

    print(f"Result shape: {result.shape}")
    print("Result:")
    print(result)
    print("SUCCESS: Memory loading works!")
