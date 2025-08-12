"""
Parquet data adapter for runtime data loading.

Handles Parquet files with optimized reading and writing for large datasets.
"""

from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from pathlib import Path

try:
    import pyarrow.parquet as pq
    import pyarrow as pa
    PYARROW_AVAILABLE = True
except ImportError:
    pq = None
    pa = None
    PYARROW_AVAILABLE = False


class ParquetAdapter:
    """
    Parquet data format adapter with optimized I/O.
    
    Features:
    - Column-oriented reading for faster variable selection
    - Predicate pushdown for time/basin filtering
    - Schema validation and conversion
    - Metadata preservation
    """
    
    @staticmethod
    def read_parquet(
        file_path: Path,
        time_column: str = "time",
        basin_column: str = "basin",
        columns: Optional[List[str]] = None,
        filters: Optional[List[Tuple]] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Read Parquet file with optional filtering and column selection.
        
        Parameters
        ----------
        file_path : Path
            Path to Parquet file
        time_column : str
            Name of time column
        basin_column : str
            Name of basin column
        columns : Optional[List[str]]
            Columns to read (None for all)
        filters : Optional[List[Tuple]]
            Filters to apply during reading
        **kwargs
            Additional parameters for pandas.read_parquet
            
        Returns
        -------
        pd.DataFrame
            DataFrame with MultiIndex (time, basin)
        """
        # Use pyarrow engine if available for better performance
        engine = 'pyarrow' if PYARROW_AVAILABLE else 'auto'
        
        # Read parquet file
        df = pd.read_parquet(
            file_path,
            columns=columns,
            filters=filters,
            engine=engine,
            **kwargs
        )
        
        # Parse time column
        if time_column in df.columns:
            df[time_column] = pd.to_datetime(df[time_column])
        else:
            raise ValueError(f"Time column '{time_column}' not found")
            
        # Handle basin column
        if basin_column not in df.columns:
            df[basin_column] = "default_basin"
            
        # Set MultiIndex
        if not isinstance(df.index, pd.MultiIndex):
            df = df.set_index([time_column, basin_column])
            
        return df.sort_index()
        
    @staticmethod
    def read_parquet_filtered(
        file_path: Path,
        time_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,
        basin_ids: Optional[List[str]] = None,
        variables: Optional[List[str]] = None,
        time_column: str = "time",
        basin_column: str = "basin",
        **kwargs
    ) -> pd.DataFrame:
        """
        Read Parquet file with optimized filtering.
        
        Uses predicate pushdown to filter data at the file level
        for better performance with large files.
        
        Parameters
        ----------
        file_path : Path
            Path to Parquet file
        time_range : Optional[Tuple[pd.Timestamp, pd.Timestamp]]
            Time range filter
        basin_ids : Optional[List[str]]
            Basin IDs to include
        variables : Optional[List[str]]
            Variables to read
        time_column : str
            Time column name
        basin_column : str
            Basin column name
        **kwargs
            Additional parameters
            
        Returns
        -------
        pd.DataFrame
            Filtered DataFrame
        """
        # Build filters for predicate pushdown
        filters = []
        
        if time_range:
            start_time, end_time = time_range
            filters.extend([
                (time_column, '>=', start_time),
                (time_column, '<=', end_time)
            ])
            
        if basin_ids:
            filters.append((basin_column, 'in', basin_ids))
            
        # Determine columns to read
        columns = None
        if variables:
            columns = [time_column, basin_column] + variables
            
        return ParquetAdapter.read_parquet(
            file_path, time_column, basin_column, columns, filters, **kwargs
        )
        
    @staticmethod
    def write_parquet(
        df: pd.DataFrame,
        file_path: Path,
        compression: str = "snappy",
        preserve_index: bool = True,
        **kwargs
    ) -> None:
        """
        Write DataFrame to Parquet with optimal compression.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to write
        file_path : Path
            Output file path
        compression : str
            Compression algorithm ("snappy", "gzip", "lz4", "brotli")
        preserve_index : bool
            Whether to preserve MultiIndex
        **kwargs
            Additional parameters for pandas.to_parquet
        """
        # Use pyarrow engine for better performance
        engine = 'pyarrow' if PYARROW_AVAILABLE else 'auto'
        
        df.to_parquet(
            file_path,
            compression=compression,
            index=preserve_index,
            engine=engine,
            **kwargs
        )
        
    @staticmethod
    def get_parquet_info(file_path: Path) -> Dict[str, Any]:
        """
        Get information about a Parquet file without loading all data.
        
        Parameters
        ----------
        file_path : Path
            Path to Parquet file
            
        Returns
        -------
        Dict[str, Any]
            File information including schema, size, row count
        """
        if not PYARROW_AVAILABLE:
            # Fallback to loading file metadata
            df_sample = pd.read_parquet(file_path, nrows=1)
            return {
                "columns": list(df_sample.columns),
                "dtypes": {col: str(dtype) for col, dtype in df_sample.dtypes.items()},
                "pyarrow_available": False
            }
            
        # Use pyarrow for efficient metadata reading
        parquet_file = pq.ParquetFile(file_path)
        
        info = {
            "num_rows": parquet_file.metadata.num_rows,
            "num_columns": parquet_file.metadata.num_columns,
            "file_size_mb": file_path.stat().st_size / (1024 * 1024),
            "schema": str(parquet_file.schema),
            "columns": parquet_file.schema.names,
            "pyarrow_available": True
        }
        
        # Add column details
        column_details = {}
        for i, name in enumerate(parquet_file.schema.names):
            field = parquet_file.schema.field(i)
            column_details[name] = {
                "type": str(field.type),
                "nullable": field.nullable
            }
        info["column_details"] = column_details
        
        return info
        
    @staticmethod
    def convert_csv_to_parquet(
        csv_path: Path,
        parquet_path: Path,
        time_column: str = "time",
        basin_column: str = "basin",
        chunk_size: int = 100000,
        compression: str = "snappy",
        **csv_kwargs
    ) -> Dict[str, Any]:
        """
        Convert CSV file to Parquet with chunked processing for large files.
        
        Parameters
        ----------
        csv_path : Path
            Input CSV file path
        parquet_path : Path
            Output Parquet file path
        time_column : str
            Time column name
        basin_column : str
            Basin column name  
        chunk_size : int
            Chunk size for processing
        compression : str
            Compression algorithm
        **csv_kwargs
            Additional CSV reading parameters
            
        Returns
        -------
        Dict[str, Any]
            Conversion statistics
        """
        stats = {
            "total_rows": 0,
            "chunks_processed": 0,
            "conversion_successful": False
        }
        
        try:
            # Process CSV in chunks
            chunk_iter = pd.read_csv(csv_path, chunksize=chunk_size, **csv_kwargs)
            
            first_chunk = True
            for chunk in chunk_iter:
                # Parse time column
                if time_column in chunk.columns:
                    chunk[time_column] = pd.to_datetime(chunk[time_column])
                    
                # Add basin column if missing
                if basin_column not in chunk.columns:
                    chunk[basin_column] = "default_basin"
                    
                # Write chunk (append mode after first chunk)
                if first_chunk:
                    ParquetAdapter.write_parquet(
                        chunk, parquet_path, compression=compression
                    )
                    first_chunk = False
                else:
                    # Append to existing file (requires pyarrow)
                    if PYARROW_AVAILABLE:
                        table = pa.Table.from_pandas(chunk)
                        with pq.ParquetWriter(parquet_path, table.schema, append=True) as writer:
                            writer.write_table(table)
                    else:
                        # Fallback: read existing, concatenate, and rewrite
                        existing_df = pd.read_parquet(parquet_path)
                        combined_df = pd.concat([existing_df, chunk], ignore_index=True)
                        ParquetAdapter.write_parquet(
                            combined_df, parquet_path, compression=compression
                        )
                        
                stats["total_rows"] += len(chunk)
                stats["chunks_processed"] += 1
                
            stats["conversion_successful"] = True
            
        except Exception as e:
            stats["error"] = str(e)
            
        return stats
        
    @staticmethod
    def optimize_parquet_file(
        input_path: Path,
        output_path: Path,
        sort_columns: Optional[List[str]] = None,
        compression: str = "snappy"
    ) -> Dict[str, Any]:
        """
        Optimize Parquet file by sorting and recompressing.
        
        Parameters
        ----------
        input_path : Path
            Input Parquet file
        output_path : Path  
            Output optimized file
        sort_columns : Optional[List[str]]
            Columns to sort by for better filtering performance
        compression : str
            Compression algorithm
            
        Returns
        -------
        Dict[str, Any]
            Optimization statistics
        """
        # Read original file
        df = pd.read_parquet(input_path)
        
        original_size = input_path.stat().st_size
        
        # Sort if requested
        if sort_columns:
            available_cols = [col for col in sort_columns if col in df.columns]
            if available_cols:
                df = df.sort_values(available_cols)
                
        # Write optimized file
        ParquetAdapter.write_parquet(df, output_path, compression=compression)
        
        new_size = output_path.stat().st_size
        
        return {
            "original_size_mb": original_size / (1024 * 1024),
            "optimized_size_mb": new_size / (1024 * 1024),
            "size_reduction_pct": ((original_size - new_size) / original_size) * 100,
            "rows_processed": len(df),
            "sorted_by": sort_columns if sort_columns else None
        }