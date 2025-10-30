"""
Runtime data loading module for real-time hydrological model execution.

This module provides atomic, flexible data loading interfaces optimized for 
real-time model runs, as opposed to the batch-oriented UnifiedDataLoader 
designed for historical data training and calibration.

Key Features:
- Atomic data loading: load only specific variables and time ranges needed
- Multi-source support: files, databases, streams, memory
- Fast response: optimized for real-time performance
- Simple interface: minimal configuration required

Author: Wenyu Ouyang
Date: 2025-01-22
"""

from .runtime_data_loader import RuntimeDataLoader, load_runtime_data

__all__ = ["RuntimeDataLoader", "load_runtime_data"]