# Smarter HVAC Scheduling from Indoor Sensor Data

Project for MMA3001 (Numerical Methods and Machine Learning) 2026.

Cleans indoor environmental sensor data from a Monash building, reconstructs missing
readings, detects when the HVAC runs and when the room is occupied, and proposes a
shorter HVAC on/off schedule that saves energy while keeping occupied hours comfortable.
The project compares interpolation methods for filling gaps, fits a first-order thermal
model of how the room warms and cools, and uses it to simulate the current and proposed
schedules.

## Engineering problem

| | |
|---|---|
| **Inputs** | Raw sensor export (CSV): sensor ID, upload time and a JSON payload of readings. Variables used: CO2 (ppm), temperature (°C), humidity (% RH), every ~10 min |
| **Outputs** | Gap-free readings with reconstruction flags; daily HVAC on/off, arrival and departure times; warm-up and cool-down time constants τ (h); proposed schedule with weekly run-hours, heated degree-hours (°C·h) and % of occupied time below comfort |
| **Model** | Room temperature follows `dT/dt = (T_inf - T) / tau`, with `T_inf` and `tau` fitted separately for HVAC on (warm-up) and off (cool-down) |
| **Supplied data** | 5 sensors, May–Dec 2024 export (180k rows); one sensor (6012002000869) is live from 29 Feb to 13 Jun 2024 |
| **Not supplied** | HVAC timer settings, occupancy, set-point and energy use; these are inferred from the data (switch times from temperature, occupancy from CO2) |

Main result: switching the HVAC on 08:00–19:20 on weekdays and off at weekends cuts
run-hours by 52% and heated degree-hours by 48%, with 0.9% of occupied time below
22.5 °C (`results/schedule_comparison.csv`).

Invalid inputs are handled as follows. A missing file or columns raises `FileNotFoundError` or `ValueError`. Readings
flagged with a sensor `errorCode` are dropped. Values outside physical ranges (CO2
350–5000 ppm, 5–40 °C, 0–100 % RH) are set to missing. Frozen sensors are rejected with
`ValueError`. Missing slots are filled only between observed readings (never
extrapolated) and flagged.

## Repository layout

```
src/mma3001_project/
  load.py             parse raw CSV, detect stale sensors, clean onto a 10-min grid
  reconstruction.py   gap filling (LOCF, linear, cubic, PCHIP) and synthetic-gap validation
  patterns.py         Savitzky-Golay rate of change, daily switch/arrival detection
  thermal.py          first-order warm-up/cool-down model, least-squares fit, validation
  schedule.py         schedule simulation, occupancy, run-hours and degree-hours
notebooks/
  01_data_audit       parse, find live sensors, clean
  02_validation       choose the reconstruction method
  03_reconstruction   fill gaps
  04_patterns         HVAC and occupancy times
  05_thermal          fit and validate the thermal model
  06_schedule         current vs proposed schedule, sensitivity
  07_performance      timing and profiling, baseline vs improved
tests/                pytest suite
results/              output tables written by the notebooks
docs/                 HTML documentation (pdoc)
reports/              pytest report
```

## Setup

```
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows PowerShell
pip install -e .
```

## Data

The raw file `5EnvSensor_MayToDec2024_180kRows.csv` (350 MB) is environmental sensor
data supplied through Monash Smart Infrastructure. It is not stored in this repository
because of its size and is not covered by this repository's MIT licence. Place it in
`data/raw/` to run the notebooks.

## Run

Run the notebooks in order, 01 to 07. Each saves the files the next one reads
(`data/processed/` and `results/`).

## Tests

```
python -m pytest
```

To regenerate the test report:

```
python -m pytest --html=reports/pytest_report.html --self-contained-html
```

## Documentation

HTML documentation of the source code is generated with pdoc:

```
pdoc src/mma3001_project -o docs
```

Open `docs/index.html` in a browser.

## Licence

See `LICENSE`.
