"""
Memory-based data source for runtime data loading.

Allows loading data directly from memory objects (DataFrame, arrays, dictionaries)
for fast access during real-time model execution.

Author: Wenyu Ouyang
Date: 2025-01-22
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
from datetime import datetime
import logging

from .base_source import BaseDataSource

logger = logging.getLogger(__name__)


class MemoryDataSource(BaseDataSource):
    """
    Memory-based data source for immediate data access.

    Supports:
    - pandas DataFrame with MultiIndex (time, basin)
    - Dictionary of basin_id -> DataFrame mappings
    - NumPy arrays with metadata
    - Direct data injection for real-time scenarios
    """

    def __init__(self, source_config: Dict[str, Any]):
        """
        Initialize memory data source.

        Parameters
        ----------
        source_config : Dict[str, Any]
            Configuration with keys:
            - data: Union[pd.DataFrame, Dict, np.ndarray], the actual data
            - time_column: str, name of time column (default: 'time')
            - basin_column: str, name of basin column (default: 'basin')
            - variables: List[str], variable names (for array data)
            - basin_ids: List[str], basin IDs (for array data)
            - time_index: pd.DatetimeIndex, time index (for array data)
            - data_format: str, format hint ('dataframe', 'dict', 'array')
        """
        super().__init__(source_config)

        self.raw_data = source_config["data"]
        self.time_column = source_config.get("time_column", "time")
        self.basin_column = source_config.get("basin_column", "basin")
        self.data_format = source_config.get("data_format", "auto")

        # For array data
        self.variables = source_config.get("variables", [])
        self.basin_ids = source_config.get("basin_ids", [])
        self.time_index = source_config.get("time_index", None)

        # Process and standardize the data
        self._processed_data = None
        self._process_data()

    def _process_data(self):
        """Process raw data into standardized DataFrame format."""
        if self.data_format == "auto":
            self.data_format = self._detect_format()

        if self.data_format == "dataframe":
            self._processed_data = self._process_dataframe()
        elif self.data_format == "dict":
            self._processed_data = self._process_dict()
        elif self.data_format == "array":
            self._processed_data = self._process_array()
        else:
            raise ValueError(f"Unsupported data format: {self.data_format}")

        logger.info(
            f"Processed {self.data_format} data with shape {self._processed_data.shape}"
        )

    def _detect_format(self) -> str:
        """Automatically detect the data format."""
        if isinstance(self.raw_data, pd.DataFrame):
            return "dataframe"
        elif isinstance(self.raw_data, dict):
            return "dict"
        elif isinstance(self.raw_data, np.ndarray):
            return "array"
        else:
            raise ValueError(f"Unsupported data type: {type(self.raw_data)}")

    def _process_dataframe(self) -> pd.DataFrame:
        """Process DataFrame input."""
        df = self.raw_data.copy()

        # Ensure proper time parsing
        if self.time_column in df.columns:
            df[self.time_column] = pd.to_datetime(df[self.time_column])
        elif "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"])
            self.time_column = "time"

        # Set up MultiIndex if not already
        if not isinstance(df.index, pd.MultiIndex):
            if self.time_column in df.columns and self.basin_column in df.columns:
                df = df.set_index([self.time_column, self.basin_column])
            elif self.time_column in df.columns:
                # Single basin case
                df[self.basin_column] = "default_basin"
                df = df.set_index([self.time_column, self.basin_column])

        return df.sort_index()

    def _process_dict(self) -> pd.DataFrame:
        """Process dictionary input (basin_id -> DataFrame)."""
        all_data = []

        for basin_id, basin_data in self.raw_data.items():
            if isinstance(basin_data, pd.DataFrame):
                df = basin_data.copy()

                # Handle case where time is already the index
                if df.index.name == self.time_column or df.index.name == 'time':
                    # Time is already the index, reset it to make it a column
                    df = df.reset_index()
                    if df.columns[0] == 'time':
                        self.time_column = 'time'

                # Add basin ID if not present
                if self.basin_column not in df.columns:
                    df[self.basin_column] = basin_id

                # Ensure time column is datetime
                if self.time_column in df.columns:
                    df[self.time_column] = pd.to_datetime(df[self.time_column])
                elif "time" in df.columns:
                    df["time"] = pd.to_datetime(df["time"])
                    self.time_column = "time"

                all_data.append(df)

            elif isinstance(basin_data, dict):
                # Convert dict to DataFrame
                df = pd.DataFrame(basin_data)
                df[self.basin_column] = basin_id

                if self.time_column in df.columns:
                    df[self.time_column] = pd.to_datetime(df[self.time_column])

                all_data.append(df)

        if not all_data:
            raise ValueError("No valid data found in dictionary")

        # Combine all basins
        combined_df = pd.concat(all_data, ignore_index=True)

        # Set MultiIndex
        if (
            self.time_column in combined_df.columns
            and self.basin_column in combined_df.columns
        ):
            combined_df = combined_df.set_index([self.time_column, self.basin_column])

        return combined_df.sort_index()

    def _process_array(self) -> pd.DataFrame:
        """Process NumPy array input."""
        if not self.variables:
            raise ValueError("Variable names required for array data")

        if not self.basin_ids:
            raise ValueError("Basin IDs required for array data")

        if self.time_index is None:
            raise ValueError("Time index required for array data")

        data = self.raw_data

        # Handle different array shapes
        if data.ndim == 3:
            # Shape: (time, basin, variables)
            n_time, n_basin, n_vars = data.shape
        elif data.ndim == 2:
            # Shape: (time, variables) - single basin
            n_time, n_vars = data.shape
            n_basin = 1
            data = data.reshape(n_time, 1, n_vars)
        else:
            raise ValueError(f"Unsupported array shape: {data.shape}")

        # Create MultiIndex
        if len(self.basin_ids) == 1 and n_basin == 1:
            basin_ids = self.basin_ids * n_basin
        else:
            basin_ids = self.basin_ids

        if len(basin_ids) != n_basin:
            raise ValueError(
                f"Basin IDs length ({len(basin_ids)}) doesn't match data ({n_basin})"
            )

        if len(self.variables) != n_vars:
            raise ValueError(
                f"Variables length ({len(self.variables)}) doesn't match data ({n_vars})"
            )

        # Create index combinations
        time_index = self.time_index[:n_time]
        index_tuples = [(t, b) for t in time_index for b in basin_ids]
        multi_index = pd.MultiIndex.from_tuples(
            index_tuples, names=[self.time_column, self.basin_column]
        )

        # Reshape data to (time*basin, variables)
        reshaped_data = data.reshape(-1, n_vars)

        # Create DataFrame
        df = pd.DataFrame(reshaped_data, index=multi_index, columns=self.variables)

        return df.sort_index()

    def load_variables(
        self,
        variables: List[str],
        basin_ids: List[str],
        time_range: Tuple[Union[str, datetime], Union[str, datetime]],
        **kwargs,
    ) -> pd.DataFrame:
        """
        Load specified variables for given basins and time range.

        Parameters
        ----------
        variables : List[str]
            Variable names to load
        basin_ids : List[str]
            Basin IDs to load
        time_range : Tuple[Union[str, datetime], Union[str, datetime]]
            (start_time, end_time) tuple
        **kwargs
            Additional parameters

        Returns
        -------
        pd.DataFrame
            DataFrame with MultiIndex (time, basin) and variables as columns
        """
        # Convert time range to datetime
        start_time = (
            pd.to_datetime(time_range[0])
            if isinstance(time_range[0], str)
            else time_range[0]
        )
        end_time = (
            pd.to_datetime(time_range[1])
            if isinstance(time_range[1], str)
            else time_range[1]
        )

        df = self._processed_data

        # Filter by time range
        time_index = df.index.get_level_values(0)
        
        # Ensure time index is datetime for comparison
        if not pd.api.types.is_datetime64_any_dtype(time_index):
            # If time index is not datetime, try to convert it
            try:
                time_index = pd.to_datetime(time_index)
                # Reconstruct the dataframe with proper datetime index
                basin_index = df.index.get_level_values(1)
                new_index = pd.MultiIndex.from_arrays([time_index, basin_index], 
                                                    names=[self.time_column, self.basin_column])
                df = df.copy()
                df.index = new_index
                time_index = df.index.get_level_values(0)
            except Exception:
                raise ValueError("Cannot convert time index to datetime for comparison")
        
        time_mask = (time_index >= start_time) & (time_index <= end_time)
        df = df[time_mask]

        # Filter by basin IDs
        basin_mask = df.index.get_level_values(1).isin(basin_ids)
        df = df[basin_mask]

        # Select variables
        available_vars = [v for v in variables if v in df.columns]
        if not available_vars:
            raise ValueError(
                f"None of the requested variables {variables} found in data"
            )

        missing_vars = [v for v in variables if v not in available_vars]
        if missing_vars:
            logger.warning(f"Variables not found: {missing_vars}")

        return df[available_vars].copy()

    def get_available_variables(
        self, basin_ids: Optional[List[str]] = None
    ) -> List[str]:
        """Get list of available variables."""
        return self._processed_data.columns.tolist()

    def get_available_basins(self) -> List[str]:
        """Get list of available basin IDs."""
        return self._processed_data.index.get_level_values(1).unique().tolist()

    def get_time_range(
        self, basin_ids: Optional[List[str]] = None
    ) -> Tuple[datetime, datetime]:
        """Get available time range."""
        time_values = self._processed_data.index.get_level_values(0)
        
        # Ensure time values are datetime for min/max operations
        if not pd.api.types.is_datetime64_any_dtype(time_values):
            time_values = pd.to_datetime(time_values)
            
        return time_values.min(), time_values.max()

    def update_data(self, new_data: Any, append: bool = False):
        """
        Update the memory data source with new data.

        Parameters
        ----------
        new_data : Any
            New data to add (same format as original)
        append : bool
            Whether to append to existing data or replace it
        """
        if append:
            # Store current data
            old_data = self._processed_data

            # Process new data
            old_raw_data = self.raw_data
            self.raw_data = new_data
            self._process_data()
            new_processed = self._processed_data

            # Combine data
            self._processed_data = pd.concat([old_data, new_processed]).sort_index()

            # Remove duplicates if any (keep last occurrence)
            self._processed_data = self._processed_data[
                ~self._processed_data.index.duplicated(keep="last")
            ]

        else:
            # Replace data
            self.raw_data = new_data
            self._process_data()

        logger.info(
            f"Updated memory data source. New shape: {self._processed_data.shape}"
        )

    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about the memory data source."""
        metadata = super().get_metadata()

        metadata.update(
            {
                "data_format": self.data_format,
                "raw_data_type": str(type(self.raw_data).__name__),
                "data_shape": self._processed_data.shape,
                "memory_usage_mb": self._processed_data.memory_usage(deep=True).sum()
                / (1024 * 1024),
                "time_column": self.time_column,
                "basin_column": self.basin_column,
            }
        )

        return metadata
