"""
JSON data adapter for runtime data loading.

Handles JSON files with various structures for hydrological data.
"""

from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime


class JSONAdapter:
    """
    JSON data format adapter with flexible structure handling.
    
    Supports various JSON structures:
    - Array of records: [{"time": "2024-01-01", "basin": "001", "prcp": 5.2}, ...]
    - Nested structure: {"basin_001": {"2024-01-01": {"prcp": 5.2, "flow": 10.1}}}
    - Time series structure: {"time": [...], "basin": [...], "prcp": [...]}
    """
    
    @staticmethod
    def read_json(
        file_path: Path,
        time_column: str = "time",
        basin_column: str = "basin",
        structure: str = "auto",
        date_format: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Read JSON file with automatic structure detection.
        
        Parameters
        ----------
        file_path : Path
            Path to JSON file
        time_column : str
            Name of time column
        basin_column : str
            Name of basin column
        structure : str
            JSON structure type ("auto", "records", "nested", "columns")
        date_format : Optional[str]
            Date format for parsing
        **kwargs
            Additional json loading parameters
            
        Returns
        -------
        pd.DataFrame
            Standardized DataFrame with MultiIndex (time, basin)
        """
        # Load JSON data
        with open(file_path, 'r', **kwargs) as f:
            data = json.load(f)
            
        # Detect structure if auto
        if structure == "auto":
            structure = JSONAdapter._detect_structure(data, time_column, basin_column)
            
        # Convert based on detected structure
        if structure == "records":
            return JSONAdapter._from_records(data, time_column, basin_column, date_format)
        elif structure == "nested":
            return JSONAdapter._from_nested(data, time_column, basin_column, date_format)
        elif structure == "columns":
            return JSONAdapter._from_columns(data, time_column, basin_column, date_format)
        else:
            raise ValueError(f"Unsupported JSON structure: {structure}")
            
    @staticmethod
    def _detect_structure(data: Any, time_column: str, basin_column: str) -> str:
        """Detect JSON structure type."""
        if isinstance(data, list):
            if data and isinstance(data[0], dict):
                # Array of records
                return "records"
        elif isinstance(data, dict):
            # Check if top-level keys look like basin IDs
            sample_keys = list(data.keys())[:5]
            if all(isinstance(data.get(key), dict) for key in sample_keys):
                return "nested"
            # Check if keys are column names
            elif time_column in data or basin_column in data:
                return "columns"
                
        # Default fallback
        return "records"
        
    @staticmethod
    def _from_records(
        data: List[Dict],
        time_column: str,
        basin_column: str,
        date_format: Optional[str]
    ) -> pd.DataFrame:
        """Convert from array of records format."""
        df = pd.DataFrame(data)
        
        # Parse time column
        if time_column in df.columns:
            df[time_column] = pd.to_datetime(df[time_column], format=date_format)
        else:
            raise ValueError(f"Time column '{time_column}' not found")
            
        # Handle basin column
        if basin_column not in df.columns:
            df[basin_column] = "default_basin"
            
        # Set MultiIndex
        df = df.set_index([time_column, basin_column])
        
        return df.sort_index()
        
    @staticmethod
    def _from_nested(
        data: Dict[str, Dict],
        time_column: str,
        basin_column: str,
        date_format: Optional[str]
    ) -> pd.DataFrame:
        """Convert from nested basin -> time -> variables format."""
        all_records = []
        
        for basin_id, basin_data in data.items():
            if isinstance(basin_data, dict):
                for time_str, variables in basin_data.items():
                    record = {
                        time_column: time_str,
                        basin_column: basin_id
                    }
                    if isinstance(variables, dict):
                        record.update(variables)
                    else:
                        # Single value case
                        record["value"] = variables
                        
                    all_records.append(record)
                    
        return JSONAdapter._from_records(all_records, time_column, basin_column, date_format)
        
    @staticmethod
    def _from_columns(
        data: Dict[str, List],
        time_column: str,
        basin_column: str,
        date_format: Optional[str]
    ) -> pd.DataFrame:
        """Convert from column-oriented format."""
        df = pd.DataFrame(data)
        
        # Parse time column
        if time_column in df.columns:
            df[time_column] = pd.to_datetime(df[time_column], format=date_format)
        else:
            raise ValueError(f"Time column '{time_column}' not found")
            
        # Handle basin column
        if basin_column not in df.columns:
            df[basin_column] = "default_basin"
            
        # Set MultiIndex
        df = df.set_index([time_column, basin_column])
        
        return df.sort_index()
        
    @staticmethod
    def write_json(
        df: pd.DataFrame,
        file_path: Path,
        structure: str = "records",
        date_format: str = "%Y-%m-%d %H:%M:%S",
        **kwargs
    ) -> None:
        """
        Write DataFrame to JSON in specified structure.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with MultiIndex (time, basin)
        file_path : Path
            Output file path
        structure : str
            Output structure ("records", "nested", "columns")
        date_format : str
            Date format for output
        **kwargs
            Additional json.dump parameters
        """
        if structure == "records":
            data = JSONAdapter._to_records(df, date_format)
        elif structure == "nested":
            data = JSONAdapter._to_nested(df, date_format)
        elif structure == "columns":
            data = JSONAdapter._to_columns(df, date_format)
        else:
            raise ValueError(f"Unsupported structure: {structure}")
            
        # Write to file
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2, default=str, **kwargs)
            
    @staticmethod
    def _to_records(df: pd.DataFrame, date_format: str) -> List[Dict]:
        """Convert DataFrame to array of records."""
        df_reset = df.reset_index()
        
        # Format time column
        time_col = df_reset.columns[0]
        if pd.api.types.is_datetime64_any_dtype(df_reset[time_col]):
            df_reset[time_col] = df_reset[time_col].dt.strftime(date_format)
            
        return df_reset.to_dict('records')
        
    @staticmethod  
    def _to_nested(df: pd.DataFrame, date_format: str) -> Dict[str, Dict]:
        """Convert DataFrame to nested basin -> time -> variables structure."""
        nested_data = {}
        
        for (time_val, basin_id), row in df.iterrows():
            if basin_id not in nested_data:
                nested_data[basin_id] = {}
                
            # Format time
            if isinstance(time_val, datetime):
                time_str = time_val.strftime(date_format)
            else:
                time_str = str(time_val)
                
            # Add variables
            nested_data[basin_id][time_str] = row.to_dict()
            
        return nested_data
        
    @staticmethod
    def _to_columns(df: pd.DataFrame, date_format: str) -> Dict[str, List]:
        """Convert DataFrame to column-oriented structure."""
        df_reset = df.reset_index()
        
        # Format time column
        time_col = df_reset.columns[0]
        if pd.api.types.is_datetime64_any_dtype(df_reset[time_col]):
            df_reset[time_col] = df_reset[time_col].dt.strftime(date_format)
            
        return df_reset.to_dict('list')
        
    @staticmethod
    def read_jsonl(
        file_path: Path,
        time_column: str = "time",
        basin_column: str = "basin",
        date_format: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Read JSON Lines (JSONL) file.
        
        Parameters
        ----------
        file_path : Path
            Path to JSONL file
        time_column : str
            Time column name
        basin_column : str
            Basin column name
        date_format : Optional[str]
            Date format for parsing
        **kwargs
            Additional parameters
            
        Returns
        -------
        pd.DataFrame
            Standardized DataFrame
        """
        records = []
        
        with open(file_path, 'r') as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    records.append(record)
                    
        return JSONAdapter._from_records(records, time_column, basin_column, date_format)
        
    @staticmethod
    def write_jsonl(
        df: pd.DataFrame,
        file_path: Path,
        date_format: str = "%Y-%m-%d %H:%M:%S",
        **kwargs
    ) -> None:
        """
        Write DataFrame to JSON Lines format.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to write
        file_path : Path
            Output file path
        date_format : str
            Date format for output
        **kwargs
            Additional parameters
        """
        records = JSONAdapter._to_records(df, date_format)
        
        with open(file_path, 'w') as f:
            for record in records:
                json.dump(record, f, default=str, **kwargs)
                f.write('\n')
                
    @staticmethod
    def validate_json_structure(
        file_path: Path,
        expected_columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Validate JSON file structure and content.
        
        Parameters
        ----------
        file_path : Path
            Path to JSON file
        expected_columns : Optional[List[str]]
            Expected column names
            
        Returns
        -------
        Dict[str, Any]
            Validation results
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            result = {
                "valid": True,
                "structure_type": JSONAdapter._detect_structure(data, "time", "basin"),
                "file_size_mb": file_path.stat().st_size / (1024 * 1024)
            }
            
            # Get sample of data structure
            if isinstance(data, list) and data:
                result["sample_record"] = data[0]
                result["total_records"] = len(data)
            elif isinstance(data, dict):
                result["top_level_keys"] = list(data.keys())[:10]
                
            # Validate expected columns if provided
            if expected_columns:
                if isinstance(data, list) and data:
                    available_cols = set(data[0].keys()) if isinstance(data[0], dict) else set()
                elif isinstance(data, dict):
                    available_cols = set(data.keys())
                else:
                    available_cols = set()
                    
                missing_cols = set(expected_columns) - available_cols
                result["missing_columns"] = list(missing_cols)
                result["has_all_columns"] = len(missing_cols) == 0
                
        except Exception as e:
            result = {
                "valid": False,
                "error": str(e)
            }
            
        return result