"""Load and clean Monash Smart Infrastructure environmental sensor data.

The raw export has one row per *upload*, not per *measurement*: each gateway
re-sends a sensor's last cached reading roughly every 10.7 minutes, whether or
not a new reading exists. The real measurement time is the ``time`` field
inside ``jsondata`` (Unix seconds, UTC); ``createdate`` is only the upload time.

Typical use::

    from mma3001_project.load import parse_env_csv, sensor_summary, clean_sensor_frame

    long = parse_env_csv("data/raw/5EnvSensor_MayToDec2024_180kRows.csv")
    print(sensor_summary(long))
    clean = clean_sensor_frame(long, "6012002000869", end="2024-06-14")
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

#: Physically plausible indoor CO2 range (ppm). Outdoor air is ~420 ppm, so
#: indoor readings well below 350 indicate a sensor fault.
CO2_VALID_RANGE = (350.0, 5000.0)

#: Minimum ratio of unique measurement times to uploaded rows for a sensor to
#: count as "live". Frozen sensors repeat one reading thousands of times, so
#: their ratio is close to zero; live sensors sit near 0.5.
MIN_UNIQUE_RATIO = 0.1

REQUIRED_COLUMNS = {"sensorid", "jsondata", "createdate"}


def parse_env_csv(path: str | Path, dedupe_payloads: bool = True) -> pd.DataFrame:
    """Flatten the raw environmental CSV into one row per reading.

    Parameters
    ----------
    path : str or Path
        Path to the raw export (e.g. ``5EnvSensor_MayToDec2024_180kRows.csv``).
    dedupe_payloads : bool, default True
        Parse each distinct (sensor, payload) pair only once. Gateways re-send
        identical payloads, so this skips most ``json.loads`` calls without
        changing the result. ``False`` parses every row (slower
        baseline).

    Returns
    -------
    pandas.DataFrame
        Long-format table with columns ``sensorid`` (str), ``time_utc``
        (tz-aware UTC timestamp of the measurement), ``variable`` (str),
        ``unit`` (str) and ``value`` (float). Readings carrying a sensor
        ``errorCode`` are dropped. Duplicate (sensor, time, variable) rows
        produced by repeated uploads are removed. ``attrs["n_upload_rows"]``
        records the raw row count per sensor for stale-sensor detection.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If required columns are missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    # dtype=str keeps the 13-digit sensor IDs intact (Excel/float would
    # round them to 6.01E+12 and make all sensors indistinguishable).
    raw = pd.read_csv(path, dtype=str)
    missing = REQUIRED_COLUMNS - set(raw.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    upload_counts = raw["sensorid"].value_counts().to_dict()
    if dedupe_payloads:
        raw = raw.drop_duplicates(["sensorid", "jsondata"])

    records = []
    for sensor_id, payload in zip(raw["sensorid"], raw["jsondata"]):
        for item in json.loads(payload):
            var = item["variable"]
            for reading in item["values"]:
                # errorCode marks a reading the sensor itself flagged invalid.
                if "errorCode" in reading or "value" not in reading:
                    continue
                records.append((sensor_id, reading["time"], var["name"],
                                var.get("unit"), float(reading["value"])))

    long = pd.DataFrame(records, columns=["sensorid", "t", "variable", "unit", "value"])
    long = long.drop_duplicates(["sensorid", "t", "variable"])
    long["time_utc"] = pd.to_datetime(long["t"], unit="s", utc=True)
    long = long.drop(columns="t").sort_values(["sensorid", "time_utc"]).reset_index(drop=True)
    long.attrs["n_upload_rows"] = upload_counts
    return long


def sensor_summary(long: pd.DataFrame) -> pd.DataFrame:
    """Summarise each sensor's coverage and flag frozen (stale) sensors.

    Parameters
    ----------
    long : pandas.DataFrame
        Output of :func:`parse_env_csv`.

    Returns
    -------
    pandas.DataFrame
        One row per sensor: uploaded rows, unique measurement times, their
        ratio, first/last measurement time, number of distinct CO2 values and
        a boolean ``is_live`` (ratio >= :data:`MIN_UNIQUE_RATIO`).
    """
    uploads = long.attrs.get("n_upload_rows", {})
    co2 = long[long["variable"] == "Carbon dioxide"]
    out = co2.groupby("sensorid").agg(
        n_readings=("time_utc", "nunique"),
        first=("time_utc", "min"),
        last=("time_utc", "max"),
        n_distinct_co2=("value", "nunique"),
    )
    out.insert(0, "n_uploads", out.index.map(uploads))
    out["unique_ratio"] = (out["n_readings"] / out["n_uploads"]).round(3)
    out["is_live"] = out["unique_ratio"] >= MIN_UNIQUE_RATIO
    return out


def load_sensor_series(
    long: pd.DataFrame,
    sensor_id: str,
    variable: str = "Carbon dioxide",
    freq: str = "10min",
    start: str | None = None,
    end: str | None = None,
    valid_range: tuple[float, float] | None = CO2_VALID_RANGE,
) -> pd.Series:
    """Return one sensor variable on a regular time grid with NaN for gaps.

    Parameters
    ----------
    long : pandas.DataFrame
        Output of :func:`parse_env_csv`.
    sensor_id : str
        13-digit sensor ID as a string, e.g. ``"6012002000869"``.
    variable : str, default "Carbon dioxide"
        Variable name as it appears in the JSON payload.
    freq : str, default "10min"
        Grid spacing (pandas offset alias). Readings are averaged into bins.
    start, end : str, optional
        UTC bounds (inclusive start, exclusive end) to trim the series to the
        sensor's live period.
    valid_range : (float, float) or None
        Values outside this range are set to NaN (treated as missing). Use
        ``None`` to skip the check for variables other than CO2.

    Returns
    -------
    pandas.Series
        Float series indexed by tz-aware UTC timestamps at spacing ``freq``.
        Missing bins are NaN. Name is ``variable``.

    Raises
    ------
    ValueError
        If the sensor is unknown, has no data for ``variable``, or is stale.
    """
    summary = sensor_summary(long)
    if sensor_id not in summary.index:
        raise ValueError(f"Unknown sensor {sensor_id!r}; available: {list(summary.index)}")
    if not summary.loc[sensor_id, "is_live"]:
        raise ValueError(
            f"Sensor {sensor_id} is stale (unique_ratio="
            f"{summary.loc[sensor_id, 'unique_ratio']}); it repeats a frozen reading."
        )
    sel = long[(long["sensorid"] == sensor_id) & (long["variable"] == variable)]
    if sel.empty:
        raise ValueError(f"No {variable!r} data for sensor {sensor_id}")

    s = sel.set_index("time_utc")["value"]
    if start is not None:
        s = s[s.index >= pd.Timestamp(start, tz="UTC")]
    if end is not None:
        s = s[s.index < pd.Timestamp(end, tz="UTC")]
    if valid_range is not None:
        lo, hi = valid_range
        s = s.where((s >= lo) & (s <= hi))
    grid = s.resample(freq).mean()
    grid.name = variable
    return grid


#: Physically plausible indoor ranges used to reject faulty readings.
VALID_RANGES = {
    "Carbon dioxide": (350.0, 5000.0),   # ppm
    "Temperature": (5.0, 40.0),          # deg C, occupied building
    "Humidity": (0.0, 100.0),            # % RH, physical limit
}

#: Short, unit-carrying column names for the cleaned table.
SHORT_NAMES = {"Carbon dioxide": "co2_ppm", "Temperature": "temp_c", "Humidity": "rh_pct"}


def clean_sensor_frame(
    long: pd.DataFrame,
    sensor_id: str,
    variables: tuple[str, ...] = ("Carbon dioxide", "Temperature", "Humidity"),
    freq: str = "10min",
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Cleaned multi-variable table for one sensor on a shared regular grid.

    Each variable goes through :func:`load_sensor_series` with its range from
    :data:`VALID_RANGES`, then all are aligned on one time grid so a missing
    reading is NaN in every column at the same timestamp.

    Parameters
    ----------
    long : pandas.DataFrame
        Output of :func:`parse_env_csv`.
    sensor_id : str
        13-digit sensor ID as a string.
    variables : tuple of str
        JSON variable names to include.
    freq, start, end
        As in :func:`load_sensor_series`.

    Returns
    -------
    pandas.DataFrame
        Columns named by :data:`SHORT_NAMES` (e.g. ``co2_ppm``, ``temp_c``,
        ``rh_pct``), tz-aware UTC index at spacing ``freq``.

    Raises
    ------
    ValueError
        As for :func:`load_sensor_series` (unknown or stale sensor, no data).
    """
    cols = {}
    for var in variables:
        s = load_sensor_series(long, sensor_id, variable=var, freq=freq, start=start,
                               end=end, valid_range=VALID_RANGES.get(var))
        cols[SHORT_NAMES.get(var, var)] = s
    return pd.DataFrame(cols)


def cleaning_report(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-column data-quality summary of a cleaned frame.

    Returns
    -------
    pandas.DataFrame
        For each column: slots, missing count and %, number of gaps, longest
        gap (samples), min, median and max.
    """
    rows = {}
    for c in frame.columns:
        s = frame[c]
        g = gap_lengths(s)
        rows[c] = {"slots": len(s), "missing": int(s.isna().sum()),
                   "missing_pct": round(100 * s.isna().mean(), 1), "n_gaps": len(g),
                   "longest_gap": int(g.max()) if len(g) else 0,
                   "min": s.min(), "median": s.median(), "max": s.max()}
    return pd.DataFrame(rows).T


def gap_lengths(series: pd.Series) -> pd.Series:
    """Lengths (in samples) of every run of consecutive NaNs in ``series``.

    Parameters
    ----------
    series : pandas.Series
        Regularly spaced series, e.g. from :func:`load_sensor_series`.

    Returns
    -------
    pandas.Series
        Integer run lengths, one entry per gap, in time order.
    """
    isna = series.isna().to_numpy()
    if not isna.any():
        return pd.Series([], dtype=int)
    # Find starts/ends of NaN runs by differencing the padded boolean mask.
    edges = np.diff(np.concatenate(([0], isna.astype(int), [0])))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return pd.Series(ends - starts, dtype=int)