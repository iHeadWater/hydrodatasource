# Runtime Data Loading System

A new atomic, flexible data loading interface optimized for **real-time hydrological model execution**, designed to complement the existing batch-oriented data loading system for training and calibration scenarios.

## Key Features

- **🔬 Atomic Interface**: Load only specific variables and time ranges needed
- **⚡ Fast Response**: Optimized for real-time performance with intelligent caching
- **🔌 Multi-Source Support**: Files (CSV, Parquet, JSON), databases (SQL), streams, memory
- **📦 Simple Usage**: One-line data loading, minimal configuration required
- **🔄 Flexible Output**: DataFrame or arrays, compatible with existing model interfaces
- **🧵 Thread-Safe**: Safe for concurrent real-time applications

## Quick Start

### Basic Usage

```python
from hydrodatasource.runtime import RuntimeDataLoader

# Initialize loader
loader = RuntimeDataLoader()

# Load data atomically
data = loader.load_variables(
    variables=["prcp", "PET", "streamflow"],
    basin_ids=["basin_001"],
    time_range=("2024-01-01", "2024-01-31"),
    source_type="csv",
    source_config={"file_path": "data.csv"}
)
```

### One-Line Loading

```python
from hydrodatasourcehydrodatasource.runtime import load_runtime_data

# Direct loading without loader instance
data = load_runtime_data(
    ["prcp", "streamflow"], 
    "basin_001",
    ("2024-01-01", "2024-01-31"),
    "csv",
    {"file_path": "data.csv"}
)
```

### Quick Methods

```python
# Quick CSV loading
data = loader.quick_load_csv(
    "data.csv", 
    ["prcp", "streamflow"], 
    "basin_001", 
    ("2024-01-01", "2024-01-31")
)

# Quick memory loading  
data = loader.quick_load_memory(
    dataframe, 
    ["prcp", "streamflow"], 
    ["basin_001"], 
    ("2024-01-01", "2024-01-31")
)

# Quick SQL loading
data = loader.quick_load_sql(
    "postgresql://user:pass@host/db",
    "hydro_data",
    ["prcp", "streamflow"], 
    "basin_001",
    ("2024-01-01", "2024-01-31")
)
```

## Supported Data Sources

### 1. File Sources

```python
# CSV files
source_config = {
    "file_path": "data.csv",
    "time_column": "time",
    "basin_column": "basin",
    "delimiter": ","
}

# Parquet files (with filtering)
source_config = {
    "file_path": "data.parquet", 
    "time_column": "timestamp",
    "basin_column": "gauge_id"
}

# JSON files  
source_config = {
    "file_path": "data.json",
    "structure": "records"  # or "nested", "columns"
}
```

### 2. SQL Databases

```python
source_config = {
    "connection_string": "postgresql://user:pass@host/db",
    "table_name": "hydro_data",
    "time_column": "timestamp", 
    "basin_column": "station_id",
    "schema": "public"
}
```

### 3. Memory Data

```python
# From DataFrame
source_config = {"data": dataframe}

# From dictionary
basin_data = {
    "basin_001": dataframe1,
    "basin_002": dataframe2
}
source_config = {"data": basin_data}

# From arrays
source_config = {
    "data": numpy_array,
    "variables": ["prcp", "pet", "flow"],
    "basin_ids": ["basin_001"],
    "time_index": pd.date_range("2024-01-01", periods=100)
}
```

### 4. Real-Time Streams

```python
from hydrodatasource.runtime.data_sources import StreamDataSource

# Initialize stream
stream_config = {
    "stream_type": "callback",
    "buffer_size": 10000,
    "variables": ["prcp", "flow"]
}
stream_source = StreamDataSource(stream_config)

# Push real-time data
stream_source.push_data({
    "time": datetime.now(),
    "basin": "basin_001", 
    "prcp": 5.2,
    "flow": 15.8
})

# Load from stream
data = stream_source.load_variables(
    ["prcp", "flow"], 
    ["basin_001"],
    (datetime.now() - timedelta(hours=1), datetime.now())
)
```

## Output Formats

### Standard DataFrame Format
```python
data = loader.load_variables(..., return_format="standard")
# Returns: DataFrame with MultiIndex (time, basin)
print(data.head())
#                           prcp  streamflow
# time                basin                
# 2024-01-01 00:00:00 basin_001  2.5      12.3
# 2024-01-01 00:00:00 basin_002  1.8      8.7
```

### Array Format (Model Compatible)
```python
p_and_e, qobs = loader.load_variables(..., return_format="arrays")
# p_and_e: [time, basin, features=2] - Precipitation and PET  
# qobs: [time, basin, features=1] - Observed streamflow
print(f"Input shape: {p_and_e.shape}")   # (31, 2, 2)
print(f"Output shape: {qobs.shape}")     # (31, 2, 1)
```

## Data Validation and Information

```python
# Validate request before loading
validation = loader.validate_request(
    variables=["prcp", "streamflow"],
    basin_ids=["basin_001"],
    time_range=("2024-01-01", "2024-01-31"),
    source_type="csv",
    source_config={"file_path": "data.csv"}
)

if validation["valid"]:
    data = loader.load_variables(...)
else:
    print("Issues:", validation["missing_variables"])
    print("Suggestions:", validation["suggestions"])

# Get source information
info = loader.get_source_info("csv", {"file_path": "data.csv"})
print(f"Available variables: {info['available_variables']}")
print(f"Time range: {info['time_range']}")
print(f"Total basins: {info['total_basins']}")
```

## Caching and Performance

The system includes intelligent caching for optimal performance:

```python
from hydrodatasource.runtime.cache_manager import configure_global_cache

# Configure global cache
cache = configure_global_cache(
    max_memory_mb=500,      # 500MB cache limit
    default_ttl_seconds=1800,  # 30 minutes TTL
    max_entries=50
)

# Get cache statistics
stats = cache.get_stats()
print(f"Hit rate: {stats['hit_rate_pct']:.1f}%")
print(f"Memory usage: {stats['memory_usage_mb']:.1f} MB")

# Clear cache when needed
cache.clear()
```

## Architecture Overview

```
RuntimeDataLoader
├── data_sources/           # Atomic data source implementations
│   ├── base_source.py     # Common interface
│   ├── file_source.py     # CSV, Parquet, JSON
│   ├── sql_source.py      # SQL databases
│   ├── memory_source.py   # In-memory data
│   └── stream_source.py   # Real-time streams
├── data_adapters/         # Format conversion utilities  
│   ├── csv_adapter.py
│   ├── parquet_adapter.py
│   ├── json_adapter.py
│   └── xarray_adapter.py
├── cache_manager.py       # Intelligent caching
└── runtime_data_loader.py # Main interface
```

## Comparison with UnifiedDataLoader

| Feature | RuntimeDataLoader | UnifiedDataLoader |
|---------|------------------|-------------------|
| **Use Case** | Real-time execution | Training & calibration |
| **Interface** | Atomic parameters | Configuration objects |
| **Performance** | Real-time optimized | Batch optimized |
| **Caching** | Intelligent caching | Basic caching |
| **Flexibility** | High - load anything | Medium - config-driven |
| **Setup Complexity** | Minimal | Moderate |

## Examples

See `examples/runtime_data_loading_examples.py` for comprehensive examples including:

1. **Basic CSV Loading** - Simple file-based data loading
2. **Memory Loading** - In-memory data structures
3. **Array Output** - Model-compatible format
4. **Validation** - Request validation and source info
5. **Streaming** - Real-time data simulation  
6. **Performance** - Comparison with traditional methods

Run examples:
```bash
cd hydrodatasource
uv run python examples/runtime_data_loading_examples.py
```

## Integration with Existing Models

You can use the RuntimeDataLoader to load data for existing models with hydromodel.

The RuntimeDataLoader produces data in the same format as existing systems:

```python
# Load data for XAJ model
p_and_e, qobs = load_runtime_data(
    ["prcp", "PET", "streamflow"],
    "basin_001", 
    ("2024-01-01", "2024-01-31"),
    "csv",
    {"file_path": "basin_001.csv"},
    return_format="arrays"
)

# Direct use with existing model
from hydromodel.models.xaj import xaj
from hydromodel.models.model_config import read_param_from_config

# Get model parameters (existing approach)
params = read_param_from_config(param_file)

# Run model with runtime-loaded data  
simulation = xaj(p_and_e, params, warmup_length=365)
```

## Thread Safety

All components are designed to be thread-safe for concurrent real-time applications:

```python
import threading
from concurrent.futures import ThreadPoolExecutor

def load_basin_data(basin_id):
    return loader.load_variables(
        ["prcp", "streamflow"],
        basin_id,
        ("2024-01-01", "2024-01-31"), 
        "csv",
        {"file_path": f"{basin_id}.csv"}
    )

# Concurrent loading
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(load_basin_data, f"basin_{i:03d}") 
               for i in range(1, 11)]
    results = [future.result() for future in futures]
```

## Best Practices

1. **Use caching** for repeated data access
2. **Validate requests** before loading in production
3. **Clear cache periodically** to manage memory
4. **Use appropriate data sources** for your use case
5. **Leverage quick methods** for simple scenarios
6. **Monitor performance** with cache statistics

## Dependencies

- **Required**: pandas, numpy
- **Optional**: 
  - sqlalchemy (for SQL sources)
  - pyarrow (for Parquet files)
  - xarray (for xarray compatibility)

Install optional dependencies:
```bash
uv add sqlalchemy pyarrow xarray
```

## Future Enhancements

- [ ] Redis/external cache support
- [ ] Message queue integration (RabbitMQ, Kafka)
- [ ] REST API data sources
- [ ] Data transformation pipelines
- [ ] Distributed loading across nodes