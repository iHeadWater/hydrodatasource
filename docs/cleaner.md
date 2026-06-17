# Cleaner

The `cleaner` module provides a suite of tools for cleaning and preprocessing raw hydrological time series data. Raw data from gauging stations often contains errors, gaps, and noise. This module helps to identify and correct these issues to prepare the data for analysis and modeling.

## Rainfall Cleaner

**File:** `rainfall_cleaner.py`

`RainfallCleaner` is used to validate and clean station-based rainfall data.

- **Yearly Check**: It compares the total annual rainfall from a station against a reference dataset (like ERA5-Land reanalysis) to identify stations that may be unreliable.
- **Extreme Value Check**: It flags and removes precipitation values that are physically unrealistic (e.g., 200 mm in a single hour).
- **Time Series Check**: It detects anomalies like sudden, sharp gradients in the data or periods where the sensor appears to be stuck on a constant low value.

## Reservoir Inflow Cleaner

**File:** `rsvr_inflow_cleaner.py`

`ReservoirInflowBacktrack` is a powerful tool for estimating reservoir inflow when direct measurements are not available. It uses the water balance method, based on reservoir water level, storage capacity, and outflow.

1.  **Clean Storage Data**: It first cleans the reservoir water level and storage data, removing outliers and fitting a water level-storage (Z-W) curve to ensure consistency.
2.  **Back-calculate Inflow**: It then calculates inflow using the principle: `Inflow = Outflow + Change in Storage`.
3.  **Correct Negative Values**: Since negative inflow is physically impossible, it adjusts these values while preserving the overall water balance.
4.  **Interpolate**: Finally, it interpolates the data to a consistent hourly time step.

## Streamflow Cleaner

**File:** `streamflow_cleaner.py`

`StreamflowCleaner` focuses on smoothing noisy streamflow data. This is often necessary to reduce measurement noise without distorting the underlying hydrological signal.

It offers several smoothing algorithms, including:
- Simple Moving Average (`moving_average`)
- Kalman Filter (`kalman_filter`)
- Low-pass Butterworth Filter (`lowpass_filter`)
- Fast Fourier Transform (`FFT`) and Wavelet (`wavelet`) filtering

All methods are designed to be volume-preserving, meaning the total volume of streamflow is not changed by the smoothing process.

## Water Level Cleaner

**File:** `waterlevel_cleaner.py`

`WaterlevelCleaner` is designed to fix anomalies in water level data.

- **Gradient Filter**: Its main feature is a `moving_gradient_filter` that identifies and removes data points where the water level changes at an unrealistic rate. It uses different thresholds for flood seasons and non-flood seasons to avoid removing legitimate rapid changes during high-flow events.
- **Filling Gaps**: It also provides a `rolling_fill` method to fill in missing data points based on the most frequent value (mode) in a moving window.
