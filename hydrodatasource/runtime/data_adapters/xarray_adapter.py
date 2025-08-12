"""
XArray data adapter for runtime data loading.

Handles conversion between xarray datasets and pandas DataFrames
for compatibility with existing hydrodatasource infrastructure.
"""

from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np

try:
    import xarray as xr
    XARRAY_AVAILABLE = True
except ImportError:
    xr = None
    XARRAY_AVAILABLE = False


class XArrayAdapter:
    """
    XArray data format adapter for hydrodatasource compatibility.
    
    Converts between xarray Datasets (used by hydrodatasource) and 
    pandas DataFrames (used by RuntimeDataLoader) while preserving
    metadata and coordinate information.
    """
    
    @staticmethod
    def from_xarray(
        dataset: "xr.Dataset",
        time_dim: str = "time",
        basin_dim: str = "basin"
    ) -> pd.DataFrame:
        """
        Convert xarray Dataset to pandas DataFrame with MultiIndex.
        
        Parameters
        ----------
        dataset : xr.Dataset
            Input xarray dataset
        time_dim : str
            Name of time dimension
        basin_dim : str
            Name of basin dimension
            
        Returns
        -------
        pd.DataFrame
            DataFrame with MultiIndex (time, basin) and variables as columns
        """
        if not XARRAY_AVAILABLE:
            raise ImportError("xarray is required for XArray adapter")
            
        # Convert to DataFrame - this automatically handles MultiIndex
        df = dataset.to_dataframe()
        
        # Ensure proper index names
        if df.index.names != [time_dim, basin_dim]:
            # Try to reorder index levels
            try:
                df = df.reorder_levels([time_dim, basin_dim])
            except (KeyError, ValueError):
                # If reordering fails, reset and set new index
                df = df.reset_index()
                if time_dim in df.columns and basin_dim in df.columns:
                    df = df.set_index([time_dim, basin_dim])
                else:
                    raise ValueError(f"Could not find dimensions {time_dim} and {basin_dim} in dataset")
        
        # Sort index for consistency
        df = df.sort_index()
        
        # Remove any NaN-only columns that might result from coordinate conversion
        df = df.dropna(axis=1, how='all')
        
        return df
        
    @staticmethod
    def to_xarray(
        df: pd.DataFrame,
        time_dim: str = "time", 
        basin_dim: str = "basin",
        attrs: Optional[Dict[str, Any]] = None
    ) -> "xr.Dataset":
        """
        Convert pandas DataFrame to xarray Dataset.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with MultiIndex (time, basin)
        time_dim : str
            Name for time dimension
        basin_dim : str  
            Name for basin dimension
        attrs : Optional[Dict[str, Any]]
            Global attributes to add to dataset
            
        Returns
        -------
        xr.Dataset
            XArray dataset with proper dimensions and coordinates
        """
        if not XARRAY_AVAILABLE:
            raise ImportError("xarray is required for XArray adapter")
            
        # Ensure proper index names
        df_copy = df.copy()
        if df_copy.index.names != [time_dim, basin_dim]:
            df_copy.index.names = [time_dim, basin_dim]
            
        # Convert to xarray Dataset
        dataset = df_copy.to_xarray()
        
        # Add global attributes if provided
        if attrs:
            dataset.attrs.update(attrs)
            
        # Add coordinate attributes
        if time_dim in dataset.coords:
            dataset[time_dim].attrs["long_name"] = "Time"
            dataset[time_dim].attrs["standard_name"] = "time"
            
        if basin_dim in dataset.coords:
            dataset[basin_dim].attrs["long_name"] = "Basin identifier"
            dataset[basin_dim].attrs["standard_name"] = "basin_id"
            
        return dataset
        
    @staticmethod
    def extract_time_series(
        dataset: "xr.Dataset",
        variable: str,
        basin_id: str,
        time_dim: str = "time",
        basin_dim: str = "basin"
    ) -> pd.Series:
        """
        Extract a single time series from xarray Dataset.
        
        Parameters
        ----------
        dataset : xr.Dataset
            Input dataset
        variable : str
            Variable name to extract
        basin_id : str
            Basin identifier
        time_dim : str
            Time dimension name
        basin_dim : str
            Basin dimension name
            
        Returns
        -------
        pd.Series
            Time series for the specified variable and basin
        """
        if not XARRAY_AVAILABLE:
            raise ImportError("xarray is required for XArray adapter")
            
        if variable not in dataset.data_vars:
            raise ValueError(f"Variable '{variable}' not found in dataset")
            
        # Select data for specific basin
        try:
            ts_data = dataset[variable].sel({basin_dim: basin_id})
            return ts_data.to_pandas()
        except KeyError:
            raise ValueError(f"Basin '{basin_id}' not found in dataset")
            
    @staticmethod
    def merge_datasets(
        datasets: List["xr.Dataset"],
        dim: str = "time",
        how: str = "outer"
    ) -> "xr.Dataset":
        """
        Merge multiple xarray datasets along a dimension.
        
        Parameters
        ---------- 
        datasets : List[xr.Dataset]
            List of datasets to merge
        dim : str
            Dimension to merge along  
        how : str
            How to handle index differences ("outer", "inner")
            
        Returns
        -------
        xr.Dataset
            Merged dataset
        """
        if not XARRAY_AVAILABLE:
            raise ImportError("xarray is required for XArray adapter")
            
        if not datasets:
            raise ValueError("No datasets provided for merging")
            
        if len(datasets) == 1:
            return datasets[0]
            
        # Use xarray's merge function
        try:
            merged = xr.merge(datasets, compat="override", join=how)
            return merged
        except Exception as e:
            # Fallback to concat if merge fails
            try:
                merged = xr.concat(datasets, dim=dim, join=how)
                return merged
            except Exception:
                raise ValueError(f"Failed to merge datasets: {e}")
                
    @staticmethod
    def resample_dataset(
        dataset: "xr.Dataset",
        freq: str,
        method: str = "mean",
        time_dim: str = "time"
    ) -> "xr.Dataset":
        """
        Resample dataset to different temporal frequency.
        
        Parameters
        ----------
        dataset : xr.Dataset
            Input dataset
        freq : str
            Target frequency (pandas frequency string)
        method : str
            Resampling method ("mean", "sum", "max", "min")
        time_dim : str
            Time dimension name
            
        Returns
        -------
        xr.Dataset
            Resampled dataset
        """
        if not XARRAY_AVAILABLE:
            raise ImportError("xarray is required for XArray adapter")
            
        # Get resampling function
        if method == "mean":
            resample_func = lambda x: x.mean()
        elif method == "sum":
            resample_func = lambda x: x.sum()
        elif method == "max":
            resample_func = lambda x: x.max()
        elif method == "min":
            resample_func = lambda x: x.min()
        else:
            raise ValueError(f"Unsupported resampling method: {method}")
            
        # Resample
        resampled = dataset.resample({time_dim: freq})
        return resampled.apply(resample_func)
        
    @staticmethod
    def add_variable_metadata(
        dataset: "xr.Dataset",
        variable_attrs: Dict[str, Dict[str, Any]]
    ) -> "xr.Dataset":
        """
        Add metadata attributes to variables in dataset.
        
        Parameters
        ----------
        dataset : xr.Dataset
            Input dataset
        variable_attrs : Dict[str, Dict[str, Any]]
            Variable attributes mapping
            
        Returns
        -------
        xr.Dataset
            Dataset with updated variable attributes
        """
        dataset_copy = dataset.copy()
        
        for var_name, attrs in variable_attrs.items():
            if var_name in dataset_copy.data_vars:
                dataset_copy[var_name].attrs.update(attrs)
                
        return dataset_copy
        
    @staticmethod
    def get_dataset_info(dataset: "xr.Dataset") -> Dict[str, Any]:
        """
        Get comprehensive information about an xarray Dataset.
        
        Parameters
        ----------
        dataset : xr.Dataset
            Input dataset
            
        Returns
        -------
        Dict[str, Any]
            Dataset information including dimensions, variables, coordinates, etc.
        """
        if not XARRAY_AVAILABLE:
            raise ImportError("xarray is required for XArray adapter")
            
        info = {
            "dimensions": dict(dataset.dims),
            "coordinates": list(dataset.coords.keys()),
            "data_variables": list(dataset.data_vars.keys()),
            "global_attributes": dict(dataset.attrs),
            "memory_usage_mb": float(dataset.nbytes) / (1024 * 1024)
        }
        
        # Add variable details
        var_details = {}
        for var_name in dataset.data_vars:
            var = dataset[var_name]
            var_details[var_name] = {
                "dtype": str(var.dtype),
                "shape": var.shape,
                "dimensions": var.dims,
                "attributes": dict(var.attrs)
            }
        info["variable_details"] = var_details
        
        # Add coordinate details  
        coord_details = {}
        for coord_name in dataset.coords:
            coord = dataset[coord_name]
            coord_details[coord_name] = {
                "dtype": str(coord.dtype),
                "size": coord.size,
                "range": [float(coord.min()), float(coord.max())] if coord.size > 0 else None
            }
        info["coordinate_details"] = coord_details
        
        return info