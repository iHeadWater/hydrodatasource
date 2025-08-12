"""
Runtime data loader for real-time hydrological model execution.

Provides a simplified, atomic interface for loading hydrological data
optimized for real-time model runs and operational scenarios.

Author: Wenyu Ouyang
Date: 2025-01-22
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
from datetime import datetime
import logging
from pathlib import Path

from .data_sources import (
    BaseDataSource,
    FileDataSource,
    MemoryDataSource,
    SQLDataSource,
    StreamDataSource,
    SQL_AVAILABLE,
    STREAM_AVAILABLE,
)

logger = logging.getLogger(__name__)


class RuntimeDataLoader:
    """
    Atomic, flexible data loader for real-time hydrological model execution.

    Key Features:
    - Simple interface: load only what you need, when you need it
    - Multi-source support: files, databases, streams, memory
    - Fast response: optimized for real-time performance
    - Minimal configuration: no complex config objects required
    - Atomic operations: each call is independent and self-contained
    """

    def __init__(self):
        """Initialize the runtime data loader."""
        self._active_sources: Dict[str, BaseDataSource] = {}
        self._source_cache = {}

    def load_variables(
        self,
        variables: List[str],
        basin_ids: Union[str, List[str]],
        time_range: Tuple[Union[str, datetime], Union[str, datetime]],
        source_type: str = "csv",
        source_config: Optional[Dict[str, Any]] = None,
        return_format: str = "standard",
    ) -> Union[pd.DataFrame, Tuple[np.ndarray, np.ndarray]]:
        """
        Load specified variables for given basins and time range.

        This is the main entry point for atomic data loading.

        Parameters
        ----------
        variables : List[str]
            Variable names to load (e.g., ["prcp", "PET", "streamflow"])
        basin_ids : Union[str, List[str]]
            Basin identifier(s) to load data for
        time_range : Tuple[Union[str, datetime], Union[str, datetime]]
            (start_time, end_time) for data loading
        source_type : str
            Type of data source ("csv", "parquet", "sql", "memory", "stream")
        source_config : Optional[Dict[str, Any]]
            Configuration for the data source. Common keys:
            - For files: {"file_path": "/path/to/data.csv"}
            - For SQL: {"connection_string": "...", "table_name": "..."}
            - For memory: {"data": dataframe_or_dict}
        return_format : str
            Return format ("standard" for DataFrame, "arrays" for (p_and_e, qobs))

        Returns
        -------
        Union[pd.DataFrame, Tuple[np.ndarray, np.ndarray]]
            Loaded data in requested format

        Examples
        --------
        >>> loader = RuntimeDataLoader()

        # Load from CSV file
        >>> data = loader.load_variables(
        ...     variables=["prcp", "PET", "streamflow"],
        ...     basin_ids="basin_001",
        ...     time_range=("2024-01-01", "2024-01-31"),
        ...     source_type="csv",
        ...     source_config={"file_path": "data.csv"}
        ... )

        # Load from database
        >>> data = loader.load_variables(
        ...     variables=["rainfall", "flow"],
        ...     basin_ids=["basin_001", "basin_002"],
        ...     time_range=(datetime(2024,1,1), datetime(2024,1,31)),
        ...     source_type="sql",
        ...     source_config={
        ...         "connection_string": "postgresql://user:pass@host/db",
        ...         "table_name": "hydro_data"
        ...     }
        ... )
        """
        # Normalize basin_ids to list
        if isinstance(basin_ids, str):
            basin_ids = [basin_ids]

        # Set default source config
        if source_config is None:
            source_config = {}

        # Create or get data source
        source = self._get_data_source(source_type, source_config)

        try:
            # Load data from source
            df = source.load_variables(variables, basin_ids, time_range)

            # Convert to requested format
            if return_format == "standard":
                return df
            elif return_format == "arrays":
                return self._convert_to_arrays(df, variables)
            else:
                raise ValueError(f"Unsupported return format: {return_format}")

        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise

    def quick_load_csv(
        self,
        file_path: Union[str, Path],
        variables: List[str],
        basin_ids: Union[str, List[str]],
        time_range: Tuple[Union[str, datetime], Union[str, datetime]],
        **kwargs,
    ) -> pd.DataFrame:
        """
        Quick method to load data from CSV file.

        Parameters
        ----------
        file_path : Union[str, Path]
            Path to CSV file
        variables : List[str]
            Variables to load
        basin_ids : Union[str, List[str]]
            Basin IDs to load
        time_range : Tuple[Union[str, datetime], Union[str, datetime]]
            Time range to load
        **kwargs
            Additional CSV reading parameters

        Returns
        -------
        pd.DataFrame
            Loaded data
        """
        source_config = {"file_path": str(file_path), **kwargs}
        return self.load_variables(
            variables,
            basin_ids,
            time_range,
            source_type="csv",
            source_config=source_config,
        )

    def quick_load_memory(
        self,
        data: Union[pd.DataFrame, Dict[str, Any], np.ndarray],
        variables: List[str],
        basin_ids: Union[str, List[str]],
        time_range: Tuple[Union[str, datetime], Union[str, datetime]],
        **kwargs,
    ) -> pd.DataFrame:
        """
        Quick method to load data from memory.

        Parameters
        ----------
        data : Union[pd.DataFrame, Dict, np.ndarray]
            Data to load from
        variables : List[str]
            Variables to load
        basin_ids : Union[str, List[str]]
            Basin IDs to load
        time_range : Tuple[Union[str, datetime], Union[str, datetime]]
            Time range to load
        **kwargs
            Additional parameters for memory source

        Returns
        -------
        pd.DataFrame
            Loaded data
        """
        source_config = {"data": data, **kwargs}
        return self.load_variables(
            variables,
            basin_ids,
            time_range,
            source_type="memory",
            source_config=source_config,
        )

    def quick_load_sql(
        self,
        connection_string: str,
        table_name: str,
        variables: List[str],
        basin_ids: Union[str, List[str]],
        time_range: Tuple[Union[str, datetime], Union[str, datetime]],
        **kwargs,
    ) -> pd.DataFrame:
        """
        Quick method to load data from SQL database.

        Parameters
        ----------
        connection_string : str
            Database connection string
        table_name : str
            Table name to query
        variables : List[str]
            Variables to load
        basin_ids : Union[str, List[str]]
            Basin IDs to load
        time_range : Tuple[Union[str, datetime], Union[str, datetime]]
            Time range to load
        **kwargs
            Additional SQL parameters

        Returns
        -------
        pd.DataFrame
            Loaded data
        """
        if not SQL_AVAILABLE:
            raise ImportError(
                "SQL support not available. Install with: pip install sqlalchemy"
            )

        source_config = {
            "connection_string": connection_string,
            "table_name": table_name,
            **kwargs,
        }
        return self.load_variables(
            variables,
            basin_ids,
            time_range,
            source_type="sql",
            source_config=source_config,
        )

    def _get_data_source(
        self, source_type: str, source_config: Dict[str, Any]
    ) -> BaseDataSource:
        """Get or create a data source instance."""
        # Create cache key
        cache_key = f"{source_type}_{hash(str(sorted(source_config.items())))}"

        if cache_key in self._active_sources:
            return self._active_sources[cache_key]

        # Create new data source
        if source_type in ["csv", "parquet", "json"]:
            source = FileDataSource({**source_config, "file_format": source_type})
        elif source_type == "file":
            source = FileDataSource(source_config)
        elif source_type == "memory":
            source = MemoryDataSource(source_config)
        elif source_type == "sql":
            if not SQL_AVAILABLE:
                raise ImportError("SQLAlchemy required for SQL sources")
            source = SQLDataSource(source_config)
        elif source_type == "stream":
            if not STREAM_AVAILABLE:
                # StreamDataSource is always available (no external dependencies)
                pass
            source = StreamDataSource(source_config)
        else:
            raise ValueError(f"Unsupported source type: {source_type}")

        # Initialize connection if needed
        if hasattr(source, "connect"):
            source.connect()

        # Cache the source
        self._active_sources[cache_key] = source

        return source

    def _convert_to_arrays(
        self, df: pd.DataFrame, variables: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert DataFrame to (p_and_e, qobs) arrays for model compatibility.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with MultiIndex (time, basin)
        variables : List[str]
            Variable names in order

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (p_and_e, qobs) arrays in standard format:
            - p_and_e: [time, basin, features=2] for precipitation and PET
            - qobs: [time, basin, features=1] for observed streamflow
        """
        # Determine variable mapping
        var_mapping = {
            "prcp": ["prcp", "precipitation", "P", "rainfall"],
            "pet": ["PET", "pet", "potential_evapotranspiration", "E"],
            "flow": ["streamflow", "flow", "Q", "discharge"],
        }

        # Find actual variable names
        prcp_var = self._find_variable(df.columns, var_mapping["prcp"])
        pet_var = self._find_variable(df.columns, var_mapping["pet"])
        flow_var = self._find_variable(df.columns, var_mapping["flow"])

        if not prcp_var:
            raise ValueError("Precipitation variable not found in data")

        # Get unique time and basin values
        time_values = df.index.get_level_values(0).unique().sort_values()
        basin_values = df.index.get_level_values(1).unique().sort_values()

        n_time = len(time_values)
        n_basin = len(basin_values)

        # Initialize arrays
        p_and_e = np.full((n_time, n_basin, 2), np.nan)
        qobs = np.full((n_time, n_basin, 1), np.nan)

        # Fill arrays
        for i, time_val in enumerate(time_values):
            for j, basin_val in enumerate(basin_values):
                try:
                    row_data = df.loc[(time_val, basin_val)]

                    # Precipitation
                    if prcp_var in row_data:
                        p_and_e[i, j, 0] = row_data[prcp_var]

                    # PET (use 0 if not available)
                    if pet_var and pet_var in row_data:
                        p_and_e[i, j, 1] = row_data[pet_var]
                    else:
                        p_and_e[i, j, 1] = 0.0

                    # Flow
                    if flow_var and flow_var in row_data:
                        qobs[i, j, 0] = row_data[flow_var]

                except KeyError:
                    # Missing data point - keep NaN values
                    continue

        return p_and_e, qobs

    def _find_variable(
        self, columns: pd.Index, possible_names: List[str]
    ) -> Optional[str]:
        """Find actual variable name from list of possibilities."""
        for name in possible_names:
            if name in columns:
                return name
        return None

    def validate_request(
        self,
        variables: List[str],
        basin_ids: Union[str, List[str]],
        time_range: Tuple[Union[str, datetime], Union[str, datetime]],
        source_type: str,
        source_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Validate a data loading request before execution.

        Parameters
        ----------
        variables : List[str]
            Variables to validate
        basin_ids : Union[str, List[str]]
            Basin IDs to validate
        time_range : Tuple[Union[str, datetime], Union[str, datetime]]
            Time range to validate
        source_type : str
            Source type to validate
        source_config : Dict[str, Any]
            Source configuration to validate

        Returns
        -------
        Dict[str, Any]
            Validation results with suggestions
        """
        try:
            # Normalize basin_ids
            if isinstance(basin_ids, str):
                basin_ids = [basin_ids]

            # Get data source
            source = self._get_data_source(source_type, source_config)

            # Validate with source
            return source.validate_request(variables, basin_ids, time_range)

        except Exception as e:
            return {
                "valid": False,
                "error": str(e),
                "suggestions": {
                    "check_source_config": "Verify source configuration parameters",
                    "check_dependencies": "Ensure required packages are installed",
                },
            }

    def get_source_info(
        self, source_type: str, source_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Get information about a data source.

        Parameters
        ----------
        source_type : str
            Type of data source
        source_config : Dict[str, Any]
            Source configuration

        Returns
        -------
        Dict[str, Any]
            Source metadata and capabilities
        """
        try:
            source = self._get_data_source(source_type, source_config)
            return source.get_metadata()
        except Exception as e:
            return {"error": str(e)}

    def clear_cache(self):
        """Clear all cached data sources."""
        for source in self._active_sources.values():
            if hasattr(source, "disconnect"):
                source.disconnect()

        self._active_sources.clear()
        self._source_cache.clear()
        logger.info("Cleared all cached data sources")

    def __del__(self):
        """Cleanup on deletion."""
        self.clear_cache()


# Convenience function for quick data loading
def load_runtime_data(
    variables: List[str],
    basin_ids: Union[str, List[str]],
    time_range: Tuple[Union[str, datetime], Union[str, datetime]],
    source_type: str = "csv",
    source_config: Optional[Dict[str, Any]] = None,
    return_format: str = "standard",
) -> Union[pd.DataFrame, Tuple[np.ndarray, np.ndarray]]:
    """
    Convenient function for quick runtime data loading.

    This is a stateless version of RuntimeDataLoader.load_variables()
    for one-off data loading operations.

    Parameters
    ----------
    variables : List[str]
        Variable names to load
    basin_ids : Union[str, List[str]]
        Basin identifier(s)
    time_range : Tuple[Union[str, datetime], Union[str, datetime]]
        (start_time, end_time) tuple
    source_type : str
        Data source type
    source_config : Optional[Dict[str, Any]]
        Source configuration
    return_format : str
        Return format ("standard" or "arrays")

    Returns
    -------
    Union[pd.DataFrame, Tuple[np.ndarray, np.ndarray]]
        Loaded data in requested format

    Examples
    --------
    >>> # Quick CSV loading
    >>> data = load_runtime_data(
    ...     ["prcp", "streamflow"],
    ...     "basin_001",
    ...     ("2024-01-01", "2024-01-31"),
    ...     "csv",
    ...     {"file_path": "data.csv"}
    ... )
    """
    loader = RuntimeDataLoader()
    try:
        return loader.load_variables(
            variables, basin_ids, time_range, source_type, source_config, return_format
        )
    finally:
        loader.clear_cache()
