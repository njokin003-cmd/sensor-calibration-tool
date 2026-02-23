# Sensor Calibration Tool

A pure C++17 command-line tool for sensor calibration with linear/polynomial 
fitting, weighted regression, confidence intervals, and batch processing.

## Features
- Linear & polynomial regression (any degree)
- Weighted least squares
- 95% confidence intervals (pure C++, no Boost)
- Batch mode — processes a whole folder of CSV files
- Residual error report (MAE, RMSE, Max error)

## Build

### Linux / macOS
    g++ -std=c++17 -O2 -o calibration_tool src/calibration_tool_extended.cpp

### Windows (MSYS2/MinGW)
    g++ -std=c++17 -O2 -o calibration_tool.exe src/calibration_tool_extended.cpp

## Usage

    # Basic linear fit
    ./calibration_tool data/sensor_data.csv

    # Polynomial degree 3
    ./calibration_tool data/sensor_data.csv --degree 3

    # Weighted regression
    ./calibration_tool data/sensor_weighted.csv --weighted

    # With 95% confidence intervals
    ./calibration_tool data/sensor_data.csv --ci

    # Batch process a folder of CSVs
    ./calibration_tool --batch data/ --out results/

## CSV Format

Two columns (with optional weight):

    raw,reference
    0.10,0.50
    1.05,5.20

Or with weights:

    raw,reference,weight
    0.10,0.50,1.0
    1.05,5.20,2.0
