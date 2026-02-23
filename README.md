Everything is in powershell

Step 1 — Clone your GitHub repo

cd C:\Users\YOUR_NAME\Desktop
git clone https://github.com/YOUR_USERNAME/sensor-calibration-tool.git
cd sensor-calibration-tool

Step 2 — Compile
g++ -std=c++17 -O2 -o calibration_tool.exe "Sensor/calibration_tool_extended (1).cpp"

Step 3 — Run(in powershell)
# Basic linear fit
.\calibration_tool.exe Sensor/sensor_data.csv

# Polynomial degree 2
.\calibration_tool.exe Sensor/sensor_data.csv --degree 2

# With confidence intervals
.\calibration_tool.exe Sensor/sensor_data.csv --ci

# Weighted regression
.\calibration_tool.exe Sensor/sensor_weighted.csv --weighted

# Batch mode
.\calibration_tool.exe --batch Sensor/ --out results/
