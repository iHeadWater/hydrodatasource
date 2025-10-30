"""
CSV data adapter for runtime data loading.

Handles various CSV formats and layouts commonly used in hydrology.
"""

from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from pathlib import Path


class CSVAdapter:
    """
    CSV data format adapter with automatic layout detection.
    
    Supports various hydrological CSV formats:
    - Single file with time, basin, variables
    - Wide format: time, basin1_var1, basin1_var2, basin2_var1, ...
    - Long format: time, basin, variable, value
    """
    
    @staticmethod
    def read_csv(
        file_path: Path,
        time_column: str = "time",
        basin_column: str = "basin",
        delimiter: str = ",",
        date_format: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Read CSV file with automatic format detection.
        
        Parameters
        ----------
        file_path : Path
            Path to CSV file
        time_column : str
            Name of time column
        basin_column : str
            Name of basin column
        delimiter : str
            CSV delimiter
        date_format : Optional[str]
            Date format for parsing
        **kwargs
            Additional pandas.read_csv parameters
            
        Returns
        -------
        pd.DataFrame
            Standardized DataFrame with MultiIndex (time, basin)
        """
        # Read CSV
        df = pd.read_csv(file_path, delimiter=delimiter, **kwargs)
        
        # Detect and convert format
        if CSVAdapter._is_long_format(df, time_column, basin_column):
            return CSVAdapter._convert_long_format(df, time_column, basin_column, date_format)
        elif CSVAdapter._is_wide_format(df, time_column):
            return CSVAdapter._convert_wide_format(df, time_column, basin_column, date_format)
        else:
            return CSVAdapter._convert_standard_format(df, time_column, basin_column, date_format)
            
    @staticmethod
    def _is_long_format(df: pd.DataFrame, time_column: str, basin_column: str) -> bool:
        """Check if DataFrame is in long format (time, basin, variable, value)."""
        expected_cols = {time_column, basin_column, "variable", "value"}
        return expected_cols.issubset(set(df.columns))
        
    @staticmethod
    def _is_wide_format(df: pd.DataFrame, time_column: str) -> bool:
        """Check if DataFrame is in wide format (time, basin1_var1, basin1_var2, ...)."""
        # Heuristic: check if column names contain underscores (basin_variable pattern)
        non_time_cols = [col for col in df.columns if col != time_column]
        underscore_cols = [col for col in non_time_cols if "_" in col]
        return len(underscore_cols) > len(non_time_cols) * 0.5
        
    @staticmethod
    def _convert_long_format(
        df: pd.DataFrame, 
        time_column: str, 
        basin_column: str,
        date_format: Optional[str]
    ) -> pd.DataFrame:
        """Convert long format to standard MultiIndex format."""
        # Parse time
        df[time_column] = pd.to_datetime(df[time_column], format=date_format)
        
        # Pivot to wide format
        df_wide = df.pivot_table(
            index=[time_column, basin_column],
            columns="variable",
            values="value",
            aggfunc="first"
        )
        
        # Flatten column names
        df_wide.columns.name = None
        
        return df_wide.sort_index()
        
    @staticmethod 
    def _convert_wide_format(
        df: pd.DataFrame,
        time_column: str,
        basin_column: str, 
        date_format: Optional[str]
    ) -> pd.DataFrame:
        """Convert wide format to standard MultiIndex format."""
        # Parse time
        df[time_column] = pd.to_datetime(df[time_column], format=date_format)
        
        # Melt wide format columns
        id_vars = [time_column]
        value_vars = [col for col in df.columns if col != time_column]
        
        df_melted = df.melt(
            id_vars=id_vars,
            value_vars=value_vars,
            var_name="basin_variable",
            value_name="value"
        )
        
        # Parse basin_variable column (e.g., "basin001_prcp" -> basin="basin001", variable="prcp")
        df_melted[["basin_id", "variable"]] = df_melted["basin_variable"].str.split("_", n=1, expand=True)
        
        # Pivot back to get variables as columns
        df_standard = df_melted.pivot_table(
            index=[time_column, "basin_id"],
            columns="variable", 
            values="value",
            aggfunc="first"
        )
        
        df_standard.columns.name = None
        df_standard.index.names = [time_column, basin_column]
        
        return df_standard.sort_index()
        
    @staticmethod
    def _convert_standard_format(
        df: pd.DataFrame,
        time_column: str,
        basin_column: str,
        date_format: Optional[str]
    ) -> pd.DataFrame:
        """Convert standard format to MultiIndex format."""
        # Parse time
        if time_column in df.columns:
            df[time_column] = pd.to_datetime(df[time_column], format=date_format)
        else:
            raise ValueError(f"Time column '{time_column}' not found")
            
        # Handle missing basin column
        if basin_column not in df.columns:
            # Single basin case
            df[basin_column] = "default_basin"
            
        # Set MultiIndex
        df = df.set_index([time_column, basin_column])
        
        return df.sort_index()
        
    @staticmethod
    def write_csv(
        df: pd.DataFrame,
        file_path: Path,
        format_type: str = "standard",
        **kwargs
    ) -> None:
        """
        Write DataFrame to CSV in specified format.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with MultiIndex (time, basin)
        file_path : Path
            Output file path
        format_type : str
            Output format ("standard", "wide", "long")
        **kwargs
            Additional pandas.to_csv parameters
        """
        if format_type == "standard":
            # Reset index to get time and basin as columns
            df_out = df.reset_index()
        elif format_type == "wide":
            df_out = CSVAdapter._to_wide_format(df)
        elif format_type == "long":
            df_out = CSVAdapter._to_long_format(df)
        else:
            raise ValueError(f"Unsupported format type: {format_type}")
            
        df_out.to_csv(file_path, index=False, **kwargs)
        
    @staticmethod
    def _to_wide_format(df: pd.DataFrame) -> pd.DataFrame:
        """Convert MultiIndex DataFrame to wide format."""
        # Reset index to get time and basin as columns
        df_reset = df.reset_index()
        
        # Melt to long format first
        time_col = df_reset.columns[0]  # First column should be time
        basin_col = df_reset.columns[1]  # Second column should be basin
        
        id_vars = [time_col]
        value_vars = [col for col in df_reset.columns if col not in [time_col, basin_col]]
        
        df_melted = df_reset.melt(
            id_vars=[time_col, basin_col],
            value_vars=value_vars,
            var_name="variable",
            value_name="value"
        )
        
        # Create basin_variable column
        df_melted["basin_variable"] = df_melted[basin_col] + "_" + df_melted["variable"]
        
        # Pivot to wide format
        df_wide = df_melted.pivot_table(
            index=time_col,
            columns="basin_variable",
            values="value",
            aggfunc="first"
        )
        
        df_wide.columns.name = None
        return df_wide.reset_index()
        
    @staticmethod
    def _to_long_format(df: pd.DataFrame) -> pd.DataFrame:
        """Convert MultiIndex DataFrame to long format."""
        # Reset index to get time and basin as columns
        df_reset = df.reset_index()
        
        time_col = df_reset.columns[0]
        basin_col = df_reset.columns[1]
        value_vars = [col for col in df_reset.columns if col not in [time_col, basin_col]]
        
        # Melt to long format
        df_long = df_reset.melt(
            id_vars=[time_col, basin_col],
            value_vars=value_vars,
            var_name="variable",
            value_name="value"
        )
        
        return df_long