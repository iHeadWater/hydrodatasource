"""
Stream-based data source for real-time data feeds.

Supports real-time data streams from various sources including message queues,
APIs, and real-time databases.

Author: Wenyu Ouyang  
Date: 2025-01-22
"""

from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import threading
import time
from collections import deque
from queue import Queue, Empty

from .base_source import BaseDataSource

logger = logging.getLogger(__name__)


class StreamDataSource(BaseDataSource):
    """
    Stream-based data source for real-time data feeds.
    
    Features:
    - Real-time data ingestion
    - Configurable buffer management
    - Multiple stream source support
    - Automatic data validation
    - Thread-safe operations
    """
    
    def __init__(self, source_config: Dict[str, Any]):
        """
        Initialize stream data source.
        
        Parameters
        ----------
        source_config : Dict[str, Any]
            Configuration with keys:
            - stream_type: str, type of stream ('api', 'queue', 'callback')
            - buffer_size: int, maximum buffer size (default: 10000)
            - buffer_duration: int, buffer duration in seconds (default: 3600)
            - update_interval: float, update interval in seconds (default: 1.0)
            - variables: List[str], expected variable names
            - basin_ids: List[str], expected basin IDs
            - time_column: str, name of time column (default: 'time')
            - basin_column: str, name of basin column (default: 'basin')
            - stream_config: Dict, stream-specific configuration
            - auto_start: bool, whether to start streaming immediately (default: False)
        """
        super().__init__(source_config)
        
        self.stream_type = source_config.get("stream_type", "callback")
        self.buffer_size = source_config.get("buffer_size", 10000)
        self.buffer_duration = source_config.get("buffer_duration", 3600)  # 1 hour
        self.update_interval = source_config.get("update_interval", 1.0)
        self.variables = source_config.get("variables", [])
        self.basin_ids_config = source_config.get("basin_ids", [])
        self.time_column = source_config.get("time_column", "time")
        self.basin_column = source_config.get("basin_column", "basin")
        self.stream_config = source_config.get("stream_config", {})
        
        # Initialize streaming components
        self._data_buffer = deque(maxlen=self.buffer_size)
        self._buffer_lock = threading.Lock()
        self._streaming_thread = None
        self._stop_streaming = threading.Event()
        self._is_streaming = False
        
        # Stream callback
        self._data_callback: Optional[Callable] = None
        
        if source_config.get("auto_start", False):
            self.start_streaming()
            
    def set_data_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """
        Set callback function for receiving new data.
        
        Parameters
        ----------
        callback : Callable
            Function that will be called with new data dict.
            Expected signature: callback(data_dict) where data_dict contains:
            - time: datetime or str
            - basin_id: str  
            - variables: Dict[str, float] with variable values
        """
        self._data_callback = callback
        
    def push_data(self, data: Dict[str, Any]):
        """
        Push new data point to the stream buffer.
        
        Parameters
        ----------
        data : Dict[str, Any]
            Data dictionary with keys:
            - time: datetime or str
            - basin_id: str
            - **variables: variable names and values
        """
        try:
            # Validate and process data
            processed_data = self._process_incoming_data(data)
            
            with self._buffer_lock:
                self._data_buffer.append(processed_data)
                
                # Remove old data if buffer duration exceeded
                current_time = datetime.now()
                cutoff_time = current_time - timedelta(seconds=self.buffer_duration)
                
                while (self._data_buffer and 
                       self._data_buffer[0]['parsed_time'] < cutoff_time):
                    self._data_buffer.popleft()
                    
        except Exception as e:
            logger.error(f"Failed to push data: {e}")
            
    def _process_incoming_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process and validate incoming data."""
        processed = data.copy()
        
        # Parse time
        if 'time' in data:
            time_val = data['time']
            if isinstance(time_val, str):
                processed['parsed_time'] = pd.to_datetime(time_val)
            elif isinstance(time_val, datetime):
                processed['parsed_time'] = time_val
            else:
                processed['parsed_time'] = datetime.now()
        else:
            processed['parsed_time'] = datetime.now()
            processed['time'] = processed['parsed_time']
            
        # Validate basin ID
        if self.basin_column not in data and 'basin_id' not in data:
            if len(self.basin_ids_config) == 1:
                processed[self.basin_column] = self.basin_ids_config[0]
            else:
                raise ValueError("Basin ID not provided and cannot be inferred")
        elif 'basin_id' in data and self.basin_column != 'basin_id':
            processed[self.basin_column] = data['basin_id']
            
        return processed
        
    def start_streaming(self):
        """Start the streaming thread."""
        if self._is_streaming:
            logger.warning("Streaming already started")
            return
            
        self._stop_streaming.clear()
        
        if self.stream_type == "api":
            self._streaming_thread = threading.Thread(target=self._api_streaming_loop)
        elif self.stream_type == "queue":
            self._streaming_thread = threading.Thread(target=self._queue_streaming_loop)
        elif self.stream_type == "callback":
            # For callback mode, streaming is event-driven
            self._is_streaming = True
            logger.info("Stream started in callback mode")
            return
        else:
            raise ValueError(f"Unsupported stream type: {self.stream_type}")
            
        self._streaming_thread.daemon = True
        self._streaming_thread.start()
        self._is_streaming = True
        logger.info(f"Stream started: {self.stream_type}")
        
    def stop_streaming(self):
        """Stop the streaming thread."""
        if not self._is_streaming:
            return
            
        self._stop_streaming.set()
        
        if self._streaming_thread and self._streaming_thread.is_alive():
            self._streaming_thread.join(timeout=5.0)
            
        self._is_streaming = False
        logger.info("Streaming stopped")
        
    def _api_streaming_loop(self):
        """Streaming loop for API-based data sources."""
        # This would implement API polling logic
        # For now, it's a placeholder
        logger.info("API streaming loop started (placeholder)")
        
        while not self._stop_streaming.is_set():
            try:
                # API polling logic would go here
                time.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"API streaming error: {e}")
                time.sleep(self.update_interval)
                
    def _queue_streaming_loop(self):
        """Streaming loop for queue-based data sources."""
        # This would implement message queue consumption
        # For now, it's a placeholder
        logger.info("Queue streaming loop started (placeholder)")
        
        while not self._stop_streaming.is_set():
            try:
                # Queue consumption logic would go here
                time.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"Queue streaming error: {e}")
                time.sleep(self.update_interval)
                
    def load_variables(
        self,
        variables: List[str],
        basin_ids: List[str],
        time_range: Tuple[Union[str, datetime], Union[str, datetime]],
        **kwargs
    ) -> pd.DataFrame:
        """
        Load variables from the stream buffer.
        
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
        start_time = pd.to_datetime(time_range[0]) if isinstance(time_range[0], str) else time_range[0]
        end_time = pd.to_datetime(time_range[1]) if isinstance(time_range[1], str) else time_range[1]
        
        # Extract data from buffer
        data_rows = []
        
        with self._buffer_lock:
            for data_point in self._data_buffer:
                point_time = data_point['parsed_time']
                point_basin = data_point.get(self.basin_column, 'unknown')
                
                # Filter by time and basin
                if (start_time <= point_time <= end_time and 
                    point_basin in basin_ids):
                    
                    # Extract variables
                    row_data = {
                        self.time_column: point_time,
                        self.basin_column: point_basin
                    }
                    
                    for var in variables:
                        row_data[var] = data_point.get(var, np.nan)
                        
                    data_rows.append(row_data)
                    
        if not data_rows:
            # Return empty DataFrame with correct structure
            columns = [self.time_column, self.basin_column] + variables
            df = pd.DataFrame(columns=columns)
            df[self.time_column] = pd.to_datetime(df[self.time_column])
            return df.set_index([self.time_column, self.basin_column])
            
        # Create DataFrame
        df = pd.DataFrame(data_rows)
        df = df.set_index([self.time_column, self.basin_column])
        
        # Select only requested variables
        available_vars = [v for v in variables if v in df.columns]
        if not available_vars:
            raise ValueError(f"None of the requested variables {variables} found in stream data")
            
        return df[available_vars].sort_index()
        
    def get_available_variables(self, basin_ids: Optional[List[str]] = None) -> List[str]:
        """Get list of available variables from recent stream data."""
        if self.variables:
            return self.variables.copy()
            
        # Try to infer from buffer
        variables = set()
        with self._buffer_lock:
            for data_point in list(self._data_buffer)[-100:]:  # Check last 100 points
                for key in data_point.keys():
                    if key not in ['time', 'parsed_time', self.basin_column, self.time_column]:
                        variables.add(key)
                        
        return list(variables)
        
    def get_available_basins(self) -> List[str]:
        """Get list of available basin IDs from stream data."""
        if self.basin_ids_config:
            return self.basin_ids_config.copy()
            
        # Try to infer from buffer
        basins = set()
        with self._buffer_lock:
            for data_point in self._data_buffer:
                basin_id = data_point.get(self.basin_column)
                if basin_id:
                    basins.add(basin_id)
                    
        return list(basins)
        
    def get_time_range(self, basin_ids: Optional[List[str]] = None) -> Tuple[datetime, datetime]:
        """Get available time range from stream buffer."""
        with self._buffer_lock:
            if not self._data_buffer:
                now = datetime.now()
                return now, now
                
            times = [data_point['parsed_time'] for data_point in self._data_buffer]
            return min(times), max(times)
            
    def get_buffer_info(self) -> Dict[str, Any]:
        """Get information about the current buffer state."""
        with self._buffer_lock:
            return {
                "buffer_size": len(self._data_buffer),
                "max_buffer_size": self.buffer_size,
                "is_streaming": self._is_streaming,
                "stream_type": self.stream_type,
                "buffer_time_range": self.get_time_range() if self._data_buffer else None
            }
            
    def clear_buffer(self):
        """Clear the data buffer."""
        with self._buffer_lock:
            self._data_buffer.clear()
        logger.info("Stream buffer cleared")
        
    def disconnect(self):
        """Disconnect and cleanup streaming resources."""
        self.stop_streaming()
        self.clear_buffer()
        
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about the stream data source."""
        metadata = super().get_metadata()
        
        metadata.update({
            "stream_type": self.stream_type,
            "buffer_size": self.buffer_size,
            "buffer_duration": self.buffer_duration,
            "update_interval": self.update_interval,
            "is_streaming": self._is_streaming,
            "current_buffer_size": len(self._data_buffer),
            "time_column": self.time_column,
            "basin_column": self.basin_column
        })
        
        return metadata