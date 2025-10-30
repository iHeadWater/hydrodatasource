"""
Data adapters for runtime data loading.

This module contains adapters for converting between different data formats
and standardizing data structures for the RuntimeDataLoader.
"""

from .csv_adapter import CSVAdapter
from .xarray_adapter import XArrayAdapter

# Optional imports with error handling
try:
    from .parquet_adapter import ParquetAdapter
    PARQUET_AVAILABLE = True
except ImportError:
    ParquetAdapter = None
    PARQUET_AVAILABLE = False

try:
    from .json_adapter import JSONAdapter  
    JSON_AVAILABLE = True
except ImportError:
    JSONAdapter = None
    JSON_AVAILABLE = False

__all__ = [
    "CSVAdapter",
    "XArrayAdapter", 
    "ParquetAdapter",
    "JSONAdapter",
    "PARQUET_AVAILABLE",
    "JSON_AVAILABLE"
]