"""
Base data source interface for runtime data loading.

This defines the common interface that all data sources must implement
for compatibility with the RuntimeDataLoader.

Author: Wenyu Ouyang
Date: 2025-01-22
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
from datetime import datetime


class BaseDataSource(ABC):
    """
    Abstract base class for all runtime data sources.

    This class defines the common interface that all data source implementations
    must follow to be compatible with the RuntimeDataLoader.
    """

    def __init__(self, source_config: Dict[str, Any]):
        """
        Initialize the data source with configuration.

        Parameters
        ----------
        source_config : Dict[str, Any]
            Configuration dictionary specific to the data source type.
            Common keys include:
            - connection_params: Connection parameters for databases
            - file_path: Path to data file for file sources
            - cache_enabled: Whether to enable caching (default: True)
            - **kwargs: Source-specific parameters
        """
        self.config = source_config
        self.cache_enabled = source_config.get("cache_enabled", True)
        self._connection = None
        self._metadata_cache = {}

    @abstractmethod
    def load_variables(
        self,
        variables: List[str],
        basin_ids: List[str],
        time_range: Tuple[Union[str, datetime], Union[str, datetime]],
        **kwargs,
    ) -> pd.DataFrame:
        """
        Load specified variables for given basins and time range.

        This is the core method that must be implemented by all data sources.

        Parameters
        ----------
        variables : List[str]
            List of variable names to load (e.g., ["prcp", "PET", "streamflow"])
        basin_ids : List[str]
            List of basin identifiers
        time_range : Tuple[Union[str, datetime], Union[str, datetime]]
            Start and end time for data loading
        **kwargs
            Additional parameters specific to the data source

        Returns
        -------
        pd.DataFrame
            DataFrame with MultiIndex (time, basin) and variables as columns
            Standard format: index=[time, basin], columns=[var1, var2, ...]
        """
        pass

    @abstractmethod
    def get_available_variables(
        self, basin_ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        Get list of available variables for specified basins.

        Parameters
        ----------
        basin_ids : Optional[List[str]]
            Basin IDs to check. If None, return variables for all basins.

        Returns
        -------
        List[str]
            List of available variable names
        """
        pass

    @abstractmethod
    def get_available_basins(self) -> List[str]:
        """
        Get list of available basin IDs.

        Returns
        -------
        List[str]
            List of available basin identifiers
        """
        pass

    @abstractmethod
    def get_time_range(
        self, basin_ids: Optional[List[str]] = None
    ) -> Tuple[datetime, datetime]:
        """
        Get the available time range for specified basins.

        Parameters
        ----------
        basin_ids : Optional[List[str]]
            Basin IDs to check. If None, return overall time range.

        Returns
        -------
        Tuple[datetime, datetime]
            (start_time, end_time) tuple
        """
        pass

    def validate_request(
        self,
        variables: List[str],
        basin_ids: List[str],
        time_range: Tuple[Union[str, datetime], Union[str, datetime]],
    ) -> Dict[str, Any]:
        """
        Validate a data loading request and return validation results.

        Parameters
        ----------
        variables : List[str]
            Requested variables
        basin_ids : List[str]
            Requested basin IDs
        time_range : Tuple[Union[str, datetime], Union[str, datetime]]
            Requested time range

        Returns
        -------
        Dict[str, Any]
            Validation results with keys:
            - valid: bool, whether request is valid
            - missing_variables: List[str], variables not available
            - missing_basins: List[str], basins not available
            - time_range_issues: str, time range problems if any
            - suggestions: Dict[str, Any], suggestions for fixing issues
        """
        result = {
            "valid": True,
            "missing_variables": [],
            "missing_basins": [],
            "time_range_issues": None,
            "suggestions": {},
        }

        try:
            # Check variables
            available_vars = self.get_available_variables(basin_ids)
            missing_vars = [v for v in variables if v not in available_vars]
            if missing_vars:
                result["missing_variables"] = missing_vars
                result["valid"] = False
                result["suggestions"]["available_variables"] = available_vars

            # Check basins
            available_basins = self.get_available_basins()
            missing_basins = [b for b in basin_ids if b not in available_basins]
            if missing_basins:
                result["missing_basins"] = missing_basins
                result["valid"] = False
                result["suggestions"]["available_basins"] = available_basins

            # Check time range
            if isinstance(time_range[0], str):
                start_time = pd.to_datetime(time_range[0])
            else:
                start_time = time_range[0]
            if isinstance(time_range[1], str):
                end_time = pd.to_datetime(time_range[1])
            else:
                end_time = time_range[1]

            available_start, available_end = self.get_time_range(basin_ids)
            
            # Convert to pandas Timestamp for comparison
            if not isinstance(available_start, pd.Timestamp):
                available_start = pd.Timestamp(available_start)
            if not isinstance(available_end, pd.Timestamp):
                available_end = pd.Timestamp(available_end)
            if not isinstance(start_time, pd.Timestamp):
                start_time = pd.Timestamp(start_time)  
            if not isinstance(end_time, pd.Timestamp):
                end_time = pd.Timestamp(end_time)
                
            if start_time < available_start or end_time > available_end:
                result["time_range_issues"] = (
                    f"Requested time range ({start_time}, {end_time}) "
                    f"outside available range ({available_start}, {available_end})"
                )
                result["valid"] = False
                result["suggestions"]["available_time_range"] = (
                    available_start,
                    available_end,
                )

        except Exception as e:
            result["valid"] = False
            result["time_range_issues"] = f"Validation error: {str(e)}"

        return result

    def connect(self) -> None:
        """
        Establish connection to the data source if needed.

        This method should be implemented by data sources that require
        connection establishment (e.g., databases, remote APIs).
        """
        pass

    def disconnect(self) -> None:
        """
        Close connection to the data source if needed.
        """
        if hasattr(self, "_connection") and self._connection:
            try:
                self._connection.close()
            except:
                pass
            self._connection = None

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()

    def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata information about the data source.

        Returns
        -------
        Dict[str, Any]
            Metadata dictionary with information like:
            - source_type: str, type of data source
            - total_basins: int, number of available basins
            - available_variables: List[str]
            - time_range: Tuple[datetime, datetime]
            - data_frequency: str, temporal resolution
            - **kwargs: Source-specific metadata
        """
        if not self._metadata_cache:
            try:
                self._metadata_cache = {
                    "source_type": self.__class__.__name__,
                    "total_basins": len(self.get_available_basins()),
                    "available_variables": self.get_available_variables(),
                    "time_range": self.get_time_range(),
                    "cache_enabled": self.cache_enabled,
                }
            except Exception as e:
                self._metadata_cache = {
                    "source_type": self.__class__.__name__,
                    "error": f"Could not load metadata: {str(e)}",
                }

        return self._metadata_cache.copy()
