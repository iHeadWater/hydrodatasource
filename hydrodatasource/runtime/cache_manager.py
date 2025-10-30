"""
Cache management for runtime data loading.

Provides intelligent caching strategies to optimize repeated data access
in real-time scenarios while managing memory usage effectively.

Author: Wenyu Ouyang
Date: 2025-01-22
"""

from typing import Dict, List, Optional, Any, Tuple, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import hashlib
import pickle
import logging
from pathlib import Path
import threading
import weakref
from collections import OrderedDict

logger = logging.getLogger(__name__)


class CacheEntry:
    """Individual cache entry with metadata."""
    
    def __init__(
        self,
        data: pd.DataFrame,
        cache_key: str,
        creation_time: datetime,
        access_count: int = 0,
        ttl_seconds: Optional[int] = None
    ):
        self.data = data
        self.cache_key = cache_key
        self.creation_time = creation_time
        self.last_access_time = creation_time
        self.access_count = access_count
        self.ttl_seconds = ttl_seconds
        self.memory_usage_mb = data.memory_usage(deep=True).sum() / (1024 * 1024)
        
    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        if self.ttl_seconds is None:
            return False
        return (datetime.now() - self.creation_time).total_seconds() > self.ttl_seconds
        
    def touch(self):
        """Update last access time and increment access count."""
        self.last_access_time = datetime.now()
        self.access_count += 1
        
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about this cache entry."""
        return {
            "cache_key": self.cache_key,
            "creation_time": self.creation_time,
            "last_access_time": self.last_access_time,
            "access_count": self.access_count,
            "memory_usage_mb": self.memory_usage_mb,
            "data_shape": self.data.shape,
            "ttl_seconds": self.ttl_seconds,
            "is_expired": self.is_expired()
        }


class RuntimeCacheManager:
    """
    Intelligent cache manager for runtime data loading.
    
    Features:
    - LRU (Least Recently Used) eviction
    - TTL (Time To Live) expiration
    - Memory usage monitoring
    - Thread-safe operations
    - Cache statistics and monitoring
    """
    
    def __init__(
        self,
        max_memory_mb: float = 1000.0,
        default_ttl_seconds: Optional[int] = 3600,
        max_entries: int = 100,
        cleanup_interval_seconds: int = 300
    ):
        """
        Initialize cache manager.
        
        Parameters
        ----------
        max_memory_mb : float
            Maximum memory usage in MB
        default_ttl_seconds : Optional[int]
            Default TTL for cache entries (None for no expiration)
        max_entries : int
            Maximum number of cache entries
        cleanup_interval_seconds : int
            Interval for automatic cleanup in seconds
        """
        self.max_memory_mb = max_memory_mb
        self.default_ttl_seconds = default_ttl_seconds
        self.max_entries = max_entries
        self.cleanup_interval_seconds = cleanup_interval_seconds
        
        # Thread-safe cache storage (OrderedDict for LRU)
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        
        # Statistics
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "expirations": 0,
            "total_requests": 0
        }
        
        # Automatic cleanup thread
        self._cleanup_timer = None
        self._start_cleanup_timer()
        
    def _generate_cache_key(
        self,
        variables: List[str],
        basin_ids: List[str],
        time_range: Tuple[Union[str, datetime], Union[str, datetime]],
        source_config: Dict[str, Any]
    ) -> str:
        """Generate unique cache key for data request."""
        key_data = {
            "variables": sorted(variables),
            "basin_ids": sorted(basin_ids),
            "time_range": (str(time_range[0]), str(time_range[1])),
            "source_config": str(sorted(source_config.items()))
        }
        
        key_str = str(key_data)
        return hashlib.md5(key_str.encode()).hexdigest()
        
    def get(
        self,
        variables: List[str],
        basin_ids: List[str],
        time_range: Tuple[Union[str, datetime], Union[str, datetime]],
        source_config: Dict[str, Any]
    ) -> Optional[pd.DataFrame]:
        """
        Get data from cache.
        
        Parameters
        ----------
        variables : List[str]
            Variable names
        basin_ids : List[str]
            Basin IDs
        time_range : Tuple[Union[str, datetime], Union[str, datetime]]
            Time range
        source_config : Dict[str, Any]
            Source configuration
            
        Returns
        -------
        Optional[pd.DataFrame]
            Cached data if available, None otherwise
        """
        cache_key = self._generate_cache_key(variables, basin_ids, time_range, source_config)
        
        with self._lock:
            self._stats["total_requests"] += 1
            
            if cache_key not in self._cache:
                self._stats["misses"] += 1
                return None
                
            entry = self._cache[cache_key]
            
            # Check if expired
            if entry.is_expired():
                del self._cache[cache_key]
                self._stats["expirations"] += 1
                self._stats["misses"] += 1
                logger.debug(f"Cache entry expired: {cache_key}")
                return None
                
            # Update access info and move to end (LRU)
            entry.touch()
            self._cache.move_to_end(cache_key)
            
            self._stats["hits"] += 1
            logger.debug(f"Cache hit: {cache_key}")
            
            return entry.data.copy()
            
    def put(
        self,
        data: pd.DataFrame,
        variables: List[str],
        basin_ids: List[str],
        time_range: Tuple[Union[str, datetime], Union[str, datetime]],
        source_config: Dict[str, Any],
        ttl_seconds: Optional[int] = None
    ) -> bool:
        """
        Put data into cache.
        
        Parameters
        ----------
        data : pd.DataFrame
            Data to cache
        variables : List[str]
            Variable names
        basin_ids : List[str]
            Basin IDs
        time_range : Tuple[Union[str, datetime], Union[str, datetime]]
            Time range
        source_config : Dict[str, Any]
            Source configuration
        ttl_seconds : Optional[int]
            TTL for this entry (uses default if None)
            
        Returns
        -------
        bool
            True if cached successfully, False otherwise
        """
        cache_key = self._generate_cache_key(variables, basin_ids, time_range, source_config)
        
        if ttl_seconds is None:
            ttl_seconds = self.default_ttl_seconds
            
        entry = CacheEntry(
            data=data.copy(),
            cache_key=cache_key,
            creation_time=datetime.now(),
            ttl_seconds=ttl_seconds
        )
        
        with self._lock:
            # Check if we have room
            self._ensure_capacity(entry.memory_usage_mb)
            
            # Add entry
            self._cache[cache_key] = entry
            logger.debug(f"Cached data: {cache_key}, size: {entry.memory_usage_mb:.2f} MB")
            
            return True
            
    def _ensure_capacity(self, new_entry_memory_mb: float):
        """Ensure cache has capacity for new entry."""
        current_memory = self.get_memory_usage_mb()
        
        # Remove expired entries first
        self._cleanup_expired()
        
        # If still over capacity, evict LRU entries
        while (current_memory + new_entry_memory_mb > self.max_memory_mb or
               len(self._cache) >= self.max_entries):
            
            if not self._cache:
                break
                
            # Remove least recently used entry (first in OrderedDict)
            oldest_key, oldest_entry = self._cache.popitem(last=False)
            current_memory -= oldest_entry.memory_usage_mb
            self._stats["evictions"] += 1
            logger.debug(f"Evicted LRU entry: {oldest_key}")
            
    def _cleanup_expired(self):
        """Remove expired entries."""
        expired_keys = []
        
        for key, entry in self._cache.items():
            if entry.is_expired():
                expired_keys.append(key)
                
        for key in expired_keys:
            del self._cache[key]
            self._stats["expirations"] += 1
            
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            logger.info("Cache cleared")
            
    def invalidate_pattern(self, pattern: str):
        """Invalidate cache entries matching a pattern."""
        with self._lock:
            keys_to_remove = []
            for key in self._cache.keys():
                if pattern in key:
                    keys_to_remove.append(key)
                    
            for key in keys_to_remove:
                del self._cache[key]
                
            logger.info(f"Invalidated {len(keys_to_remove)} cache entries matching pattern: {pattern}")
            
    def get_memory_usage_mb(self) -> float:
        """Get total memory usage of cache in MB."""
        with self._lock:
            return sum(entry.memory_usage_mb for entry in self._cache.values())
            
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            current_memory = self.get_memory_usage_mb()
            hit_rate = (self._stats["hits"] / self._stats["total_requests"] 
                       if self._stats["total_requests"] > 0 else 0.0)
            
            return {
                "total_entries": len(self._cache),
                "memory_usage_mb": current_memory,
                "memory_limit_mb": self.max_memory_mb,
                "memory_utilization_pct": (current_memory / self.max_memory_mb) * 100,
                "hit_rate_pct": hit_rate * 100,
                **self._stats
            }
            
    def get_entry_details(self) -> List[Dict[str, Any]]:
        """Get detailed information about all cache entries."""
        with self._lock:
            return [entry.get_metadata() for entry in self._cache.values()]
            
    def _start_cleanup_timer(self):
        """Start automatic cleanup timer."""
        self._cleanup_timer = threading.Timer(
            self.cleanup_interval_seconds,
            self._periodic_cleanup
        )
        self._cleanup_timer.daemon = True
        self._cleanup_timer.start()
        
    def _periodic_cleanup(self):
        """Periodic cleanup of expired entries."""
        try:
            with self._lock:
                self._cleanup_expired()
                
            # Restart timer
            self._start_cleanup_timer()
            
        except Exception as e:
            logger.error(f"Error during periodic cleanup: {e}")
            
    def shutdown(self):
        """Shutdown cache manager and cleanup resources."""
        if self._cleanup_timer:
            self._cleanup_timer.cancel()
        self.clear()
        
    def __del__(self):
        """Cleanup on deletion."""
        self.shutdown()


class SimpleCache:
    """
    Simple in-memory cache for basic use cases.
    
    Provides basic caching without advanced features like TTL or memory management.
    Suitable for simple scenarios with limited data.
    """
    
    def __init__(self, max_entries: int = 50):
        """
        Initialize simple cache.
        
        Parameters
        ----------
        max_entries : int
            Maximum number of cache entries
        """
        self.max_entries = max_entries
        self._cache: Dict[str, pd.DataFrame] = {}
        self._access_order: List[str] = []
        self._lock = threading.Lock()
        
    def get(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Get data from cache by key."""
        with self._lock:
            if cache_key in self._cache:
                # Move to end (most recently used)
                self._access_order.remove(cache_key)
                self._access_order.append(cache_key)
                return self._cache[cache_key].copy()
            return None
            
    def put(self, cache_key: str, data: pd.DataFrame):
        """Put data into cache."""
        with self._lock:
            # Remove oldest entry if at capacity
            if len(self._cache) >= self.max_entries and cache_key not in self._cache:
                oldest_key = self._access_order.pop(0)
                del self._cache[oldest_key]
                
            # Add/update entry
            if cache_key in self._cache:
                self._access_order.remove(cache_key)
            else:
                self._cache[cache_key] = data.copy()
                
            self._access_order.append(cache_key)
            
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
            
    def size(self) -> int:
        """Get number of cache entries."""
        return len(self._cache)


# Global cache instance for convenience
_global_cache_manager: Optional[RuntimeCacheManager] = None
_global_cache_lock = threading.Lock()


def get_global_cache() -> RuntimeCacheManager:
    """Get or create global cache manager instance."""
    global _global_cache_manager
    
    if _global_cache_manager is None:
        with _global_cache_lock:
            if _global_cache_manager is None:
                _global_cache_manager = RuntimeCacheManager()
                
    return _global_cache_manager


def configure_global_cache(**kwargs) -> RuntimeCacheManager:
    """Configure global cache manager with custom settings."""
    global _global_cache_manager
    
    with _global_cache_lock:
        if _global_cache_manager is not None:
            _global_cache_manager.shutdown()
            
        _global_cache_manager = RuntimeCacheManager(**kwargs)
        
    return _global_cache_manager


def clear_global_cache():
    """Clear global cache."""
    cache = get_global_cache()
    cache.clear()