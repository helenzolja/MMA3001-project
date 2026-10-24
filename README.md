# Smarter HVAC scheduling from indoor sensor data

MMA3001 individual project. Cleans Monash Smart Infrastructure environmental sensor
data, reconstructs missing readings, identifies the daily HVAC and occupancy patterns,
and proposes an HVAC on/off schedule that uses less energy while keeping occupied
hours comfortable.

## Results (sensor 6012002000869, 29 Feb – 13 Jun 2024)

| Stage | Method (unit week) | Key result |
|---|---|---|
| Cleaning | Range checks, stale-sensor detection | 1,391 missing 10-min slots (all 1–2 slots long) |
| Reconstruction | Linear / cubic / PCHIP / LOCF interpolation (Wk 3), synthetic-gap validation, bootstrap CI (Wk 5) | Linear chosen; June test RMSE 9.7 ppm (95% CI 9.1–10.3) vs 11.0 for LOCF |
| Patterns | Savitzky–Golay derivative (Wk 7.1) | HVAC on 07:30, off ~midnight, 7 days a week; weekday arrivals ~10:45, weekends nearly empty |
| Thermal model | First-order ODE, least-squares fit (Wk 7.2) | Warm-up τ = 0.77 h, cool-down τ = 3.69 h; June RMSE 0.59 °C vs 2.88 °C for no-change |
| Schedule | Exact exponential simulation, trapezoidal integration (Wk 6) | Weekdays 08:00–19:20, weekends off: **52% fewer run-hours, 48% fewer heated degree-hours**, 0.9% of occupied time below 22.5 °C |
| Performance | `perf_counter`, `cProfile` | Parser 1.8× faster, validation 3× faster, identical outputs |

## Installation

```
git clone https://github.com/helenzolja/MMA3001-project.git
cd MMA3001-project
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -e .
```

The raw file `5EnvSensor_MayToDec2024_180kRows.csv` (350 MB) is not in the repository
because of its size. Place it in `data/raw/`.

## Usage

Run the notebooks in order; each saves the files the next one reads.

| Notebook | Does | Saves |
|---|---|---|
| `01_data_audit` | Parse raw file, find live sensors, clean sensor 869 | `data/processed/sensor869_clean.csv` |
| `02_validation` | Compare reconstruction methods on hidden real readings | `results/reconstruction_*.csv` |
| `03_reconstruction` | Fill gaps with the chosen method | `data/processed/sensor869_reconstructed.csv` |
| `04_patterns` | Detect HVAC switch-on/off and occupancy times | `results/daily_events.csv`, summaries |
| `05_thermal` | Fit and validate warm-up/cool-down models | `results/thermal_*.csv`, `schedule_inputs.csv` |
| `06_schedule` | Simulate current vs proposed schedule | `results/schedule_*.csv` |
| `07_performance` | Time and profile baseline vs improved code | `results/performance.csv` |

## Tests and documentation

```
python -m pytest
```

- Test report: `reports/pytest_report.html`
- API documentation: `docs/index.html` (generated with pdoc)

## Structure

```
src/mma3001_project/   load.py, reconstruction.py, patterns.py, thermal.py, schedule.py
tests/                 one test file per module
notebooks/             01–07, run in order
results/               output tables
docs/                  pdoc HTML documentation
reports/               pytest report
```

## Use of AI

Claude (Anthropic) was used to help plan the methods, draft code and explain concepts.
All code was typed, run and checked by me; results were verified by the tests and the
validation steps in the notebooks. See the report for a full reflection.

## Licence

MIT – see `LICENSE`.
