"""
File-based data source for runtime data loading.

Supports various file formats including CSV, Parquet, JSON and automatically
detects and handles different hydrological data layouts.

Author: Wenyu Ouyang
Date: 2025-01-22
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
from datetime import datetime
import logging

from .base_source import BaseDataSource

logger = logging.getLogger(__name__)


class FileDataSource(BaseDataSource):
    """
    File-based data source supporting multiple formats and layouts.

    Supports:
    - Single file with multiple basins
    - Directory with one file per basin
    - Multiple files with different variables
    - CSV, Parquet, JSON formats
    - Automatic format detection
    """

    def __init__(self, source_config: Dict[str, Any]):
        """
        Initialize file data source.

        Parameters
        ----------
        source_config : Dict[str, Any]
            Configuration with keys:
            - file_path: str, path to file or directory
            - file_format: str, optional format hint ('csv', 'parquet', 'json')
            - layout: str, data layout ('single_file', 'basin_files', 'variable_files')
            - time_column: str, name of time column (default: 'time')
            - basin_column: str, name of basin ID column (default: 'basin')
            - date_format: str, date parsing format (optional)
            - delimiter: str, delimiter for CSV files (default: ',')
            - **kwargs: Additional file reading parameters
        """
        super().__init__(source_config)

        self.file_path = Path(source_config["file_path"])
        self.file_format = source_config.get("file_format", "auto")
        self.layout = source_config.get("layout", "auto")
        self.time_column = source_config.get("time_column", "time")
        self.basin_column = source_config.get("basin_column", "basin")
        self.date_format = source_config.get("date_format", None)
        self.delimiter = source_config.get("delimiter", ",")

        # Additional parameters for file reading
        self.read_kwargs = {
            k: v
            for k, v in source_config.items()
            if k
            not in [
                "file_path",
                "file_format",
                "layout",
                "time_column",
                "basin_column",
                "date_format",
                "delimiter",
                "cache_enabled",
            ]
        }

        # Initialize metadata
        self._data_cache = {}
        self._file_info = None
        self._detect_layout()

    def _detect_layout(self):
        """Automatically detect the data layout and format."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"File or directory not found: {self.file_path}")

        if self.file_path.is_file():
            # Single file
            if self.file_format == "auto":
                self.file_format = self._detect_format(self.file_path)
            self.layout = "single_file"

        elif self.file_path.is_dir():
            # Directory with multiple files
            files = list(self.file_path.glob("*.csv")) + list(
                self.file_path.glob("*.parquet")
            )
            if not files:
                raise ValueError(
                    f"No supported files found in directory: {self.file_path}"
                )

            # Detect if files are basin-based or variable-based
            if self.layout == "auto":
                self.layout = self._detect_directory_layout(files)

            if self.file_format == "auto":
                self.file_format = self._detect_format(files[0])
        else:
            raise ValueError(f"Invalid path: {self.file_path}")

        logger.info(f"Detected layout: {self.layout}, format: {self.file_format}")

    def _detect_format(self, file_path: Path) -> str:
        """Detect file format from extension."""
        suffix = file_path.suffix.lower()
        if suffix == ".csv":
            return "csv"
        elif suffix == ".parquet":
            return "parquet"
        elif suffix in [".json", ".jsonl"]:
            return "json"
        else:
            return "csv"  # Default fallback

    def _detect_directory_layout(self, files: List[Path]) -> str:
        """Detect whether directory contains basin files or variable files."""
        # Simple heuristic: check if filenames look like basin IDs
        basin_like_pattern = any(
            f.stem.replace("_", "").replace("-", "").isalnum() for f in files[:5]
        )  # Check first 5 files

        if basin_like_pattern:
            return "basin_files"
        else:
            return "variable_files"

    def _load_single_file(self) -> pd.DataFrame:
        """Load data from a single file."""
        if self.file_format == "csv":
            df = pd.read_csv(
                self.file_path, delimiter=self.delimiter, **self.read_kwargs
            )
        elif self.file_format == "parquet":
            df = pd.read_parquet(self.file_path, **self.read_kwargs)
        elif self.file_format == "json":
            df = pd.read_json(self.file_path, **self.read_kwargs)
        else:
            raise ValueError(f"Unsupported file format: {self.file_format}")

        # Parse time column
        if self.time_column in df.columns:
            df[self.time_column] = pd.to_datetime(
                df[self.time_column], format=self.date_format
            )
        else:
            raise ValueError(f"Time column '{self.time_column}' not found in data")

        return df

    def _load_basin_files(self) -> pd.DataFrame:
        """Load data from multiple basin files."""
        all_data = []
        basin_files = {}

        # Find all basin files
        for ext in ["csv", "parquet"]:
            for file_path in self.file_path.glob(f"*.{ext}"):
                basin_id = file_path.stem
                basin_files[basin_id] = file_path

        if not basin_files:
            raise ValueError("No basin files found")

        # Load each basin file
        for basin_id, file_path in basin_files.items():
            try:
                if file_path.suffix.lower() == ".csv":
                    df = pd.read_csv(
                        file_path, delimiter=self.delimiter, **self.read_kwargs
                    )
                elif file_path.suffix.lower() == ".parquet":
                    df = pd.read_parquet(file_path, **self.read_kwargs)
                else:
                    continue

                # Add basin ID if not present
                if self.basin_column not in df.columns:
                    df[self.basin_column] = basin_id

                # Parse time column
                if self.time_column in df.columns:
                    df[self.time_column] = pd.to_datetime(
                        df[self.time_column], format=self.date_format
                    )

                all_data.append(df)

            except Exception as e:
                logger.warning(f"Failed to load basin file {file_path}: {e}")
                continue

        if not all_data:
            raise ValueError("No valid basin files could be loaded")

        return pd.concat(all_data, ignore_index=True)

    def _get_full_data(self) -> pd.DataFrame:
        """Load and cache full dataset."""
        if not self._data_cache:
            if self.layout == "single_file":
                df = self._load_single_file()

                # For single files without basin column, infer basin ID from filename
                if self.basin_column not in df.columns:
                    # Try to extract basin ID from filename
                    basename = self.file_path.stem
                    # Remove common extensions and prefixes
                    for prefix in ["timeseries_", "data_", "hydro_"]:
                        if basename.startswith(prefix):
                            basename = basename[len(prefix) :]

                    # Use filename as basin ID
                    df[self.basin_column] = basename
                    logger.info(
                        f"Inferred basin ID '{basename}' from filename: {self.file_path.name}"
                    )

            elif self.layout == "basin_files":
                df = self._load_basin_files()
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")

            # Set up MultiIndex if needed
            if self.time_column in df.columns and self.basin_column in df.columns:
                df = df.set_index([self.time_column, self.basin_column]).sort_index()
            elif self.time_column in df.columns:
                # Only time column available, set as single index
                df = df.set_index(self.time_column).sort_index()

            self._data_cache = df

        return self._data_cache

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

        # Load full data
        df = self._get_full_data()

        # Filter by time range
        if isinstance(df.index, pd.MultiIndex):
            # MultiIndex case: (time, basin)
            time_mask = (df.index.get_level_values(0) >= start_time) & (
                df.index.get_level_values(0) <= end_time
            )
            df = df[time_mask]

            # Filter by basin IDs
            basin_mask = df.index.get_level_values(1).isin(basin_ids)
            df = df[basin_mask]
        else:
            # Single index case: check if index is time or something else
            if pd.api.types.is_datetime64_any_dtype(df.index):
                # Time index case
                time_mask = (df.index >= start_time) & (df.index <= end_time)
                df = df[time_mask]

                # For single-basin files, check if requested basin matches inferred basin
                if self.basin_column in df.columns:
                    # Basin column exists, filter by it
                    basin_mask = df[self.basin_column].isin(basin_ids)
                    df = df[basin_mask]
                    if df.empty:
                        logger.warning(
                            f"No data found for basins {basin_ids}. Available basin(s): {df[self.basin_column].unique().tolist()}"
                        )
                else:
                    # No basin column, this is a single-basin file
                    # The basin ID was inferred from filename, so we need to check if requested basin matches
                    logger.info(
                        f"Single-basin file detected. Requested basins: {basin_ids}"
                    )
            else:
                # Non-time index, fallback to column filtering
                if self.time_column in df.columns:
                    df = df[
                        (df[self.time_column] >= start_time)
                        & (df[self.time_column] <= end_time)
                    ]
                if self.basin_column in df.columns:
                    df = df[df[self.basin_column].isin(basin_ids)]

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
        df = self._get_full_data()

        # Exclude index columns and metadata columns
        exclude_cols = {self.time_column, self.basin_column, "index"}
        variables = [col for col in df.columns if col not in exclude_cols]

        return variables

    def get_available_basins(self) -> List[str]:
        """Get list of available basin IDs."""
        df = self._get_full_data()

        if isinstance(df.index, pd.MultiIndex):
            return df.index.get_level_values(1).unique().tolist()
        elif self.basin_column in df.columns:
            return df[self.basin_column].unique().tolist()
        else:
            # For single-basin files, return the basename from the filename
            basename = self.file_path.stem
            # Remove common prefixes if present
            for prefix in ["timeseries_", "data_", "hydro_"]:
                if basename.startswith(prefix):
                    basename = basename[len(prefix) :]
            return [basename]

    def get_time_range(
        self, basin_ids: Optional[List[str]] = None
    ) -> Tuple[datetime, datetime]:
        """Get available time range."""
        df = self._get_full_data()

        if isinstance(df.index, pd.MultiIndex):
            time_values = df.index.get_level_values(0)
        elif self.time_column in df.columns:
            time_values = df[self.time_column]
        else:
            raise ValueError("No time information found in data")

        return time_values.min(), time_values.max()

    def get_metadata(self) -> Dict[str, Any]:
        """Get detailed metadata about the file data source."""
        metadata = super().get_metadata()

        try:
            metadata.update(
                {
                    "file_path": str(self.file_path),
                    "file_format": self.file_format,
                    "layout": self.layout,
                    "file_size_mb": (
                        self.file_path.stat().st_size / (1024 * 1024)
                        if self.file_path.is_file()
                        else None
                    ),
                    "time_column": self.time_column,
                    "basin_column": self.basin_column,
                }
            )

            # Add data shape information
            df = self._get_full_data()
            metadata.update(
                {
                    "data_shape": df.shape,
                    "memory_usage_mb": df.memory_usage(deep=True).sum() / (1024 * 1024),
                }
            )

        except Exception as e:
            metadata["metadata_error"] = str(e)

        return metadata
