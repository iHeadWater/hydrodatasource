# API Reference

This page provides an auto-generated API reference for the key components of the `hydrodatasource` library.

## Data Resolver (Unified Data Interface)

::: hydrodatasource.configs.data_resolver.open_dataset
    handler: python
    options:
      show_root_heading: true
      show_source: false

::: hydrodatasource.configs.data_resolver.resolve_data_path
    handler: python
    options:
      show_root_heading: true
      show_source: false

::: hydrodatasource.configs.data_resolver.ResolverContext
    handler: python
    options:
      show_root_heading: true
      show_source: false

`READER_ALIASES` maps every reader alias to its module/class, and `HDS_DATASETS` is the in-code
registry of hydrodatasource datasets (e.g. `songliao_event`). `DatasetResolutionError` is
re-exported from hydrodataset.

## Reader

### HydroData (base class)

::: hydrodatasource.reader.data_source.HydroData
    handler: python
    options:
      show_root_heading: true
      show_source: false

### SelfMadeHydroDataset

::: hydrodatasource.reader.data_source.SelfMadeHydroDataset
    handler: python
    options:
      show_root_heading: true
      show_source: false

### Other Readers

::: hydrodatasource.reader.data_source.LongTermDataset
    handler: python
    options:
      show_root_heading: true
      show_source: false

::: hydrodatasource.reader.data_source.SelfMadeForecastDataset
    handler: python
    options:
      show_root_heading: true
      show_source: false

::: hydrodatasource.reader.data_source.StationHydroDataset
    handler: python
    options:
      show_root_heading: true
      show_source: false

::: hydrodatasource.reader.data_source.TgHydroDatasource
    handler: python
    options:
      show_root_heading: true
      show_source: false

::: hydrodatasource.reader.floodevent.FloodEventDatasource
    handler: python
    options:
      show_root_heading: true
      show_source: false

::: hydrodatasource.reader.gages.Gages
    handler: python
    options:
      show_root_heading: true
      show_source: false

::: hydrodatasource.reader.grdc.Grdc
    handler: python
    options:
      show_root_heading: true
      show_source: false

::: hydrodatasource.reader.rainfall_reader.RainfallReader
    handler: python
    options:
      show_root_heading: true
      show_source: false

::: hydrodatasource.reader.reservoir_datasets.Crd
    handler: python
    options:
      show_root_heading: true
      show_source: false

::: hydrodatasource.reader.rsvr_inflow_reader.RsvrInflowReader
    handler: python
    options:
      show_root_heading: true
      show_source: false

## Processor

### Basin Mean Rainfall

::: hydrodatasource.processor.basin_mean_rainfall.basin_mean_func
    handler: python
    options:
      show_root_heading: true
      show_source: false

### Rainfall-Runoff Event Identification

::: hydrodatasource.processor.dmca_esr.get_rr_events
    handler: python
    options:
      show_root_heading: true
      show_source: false

## Cleaner

### Cleaner (base class)

::: hydrodatasource.cleaner.cleaner.Cleaner
    handler: python
    options:
      show_root_heading: true
      show_source: false

### RainfallCleaner

::: hydrodatasource.cleaner.rainfall_cleaner.RainfallCleaner
    handler: python
    options:
      show_root_heading: true
      show_source: false

### ReservoirInflowBacktrack

::: hydrodatasource.cleaner.rsvr_inflow_cleaner.ReservoirInflowBacktrack
    handler: python
    options:
      show_root_heading: true
      show_source: false

### StreamflowCleaner

::: hydrodatasource.cleaner.streamflow_cleaner.StreamflowCleaner
    handler: python
    options:
      show_root_heading: true
      show_source: false

### WaterlevelCleaner

::: hydrodatasource.cleaner.waterlevel_cleaner.WaterlevelCleaner
    handler: python
    options:
      show_root_heading: true
      show_source: false
