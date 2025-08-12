"""
Data sources for runtime data loading.

This module contains various data source implementations that can be used
with the RuntimeDataLoader for different input types.
"""

from .base_source import BaseDataSource
from .file_source import FileDataSource
from .memory_source import MemoryDataSource

# Optional imports with error handling
try:
    from .sql_source import SQLDataSource
    SQL_AVAILABLE = True
except ImportError:
    SQLDataSource = None
    SQL_AVAILABLE = False

try:
    from .stream_source import StreamDataSource  
    STREAM_AVAILABLE = True
except ImportError:
    StreamDataSource = None
    STREAM_AVAILABLE = False

__all__ = [
    "BaseDataSource",
    "FileDataSource", 
    "MemoryDataSource",
    "SQLDataSource",
    "StreamDataSource",
    "SQL_AVAILABLE",
    "STREAM_AVAILABLE"
]