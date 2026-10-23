"""Daily and weekly operating patterns from reconstructed sensor data.

HVAC switch-on and switch-off show up as the fastest daily rise and fall in
temperature; people arriving and leaving show up as the fastest daily rise
and fall in CO2. This module estimates those rates of change with a
smoothed finite-difference derivative (Week 7.1) and finds the time of the
strongest event in each day.

Example
-------
>>> from mma3001_project.patterns import rate_of_change, daily_event
>>> dT = rate_of_change(recon["temp_c"])
>>> hvac_on = daily_event(dT, start_hour=4, end_hour=12, kind="max", min_rate=0.5)
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

LOCAL_TZ = "Australia/Melbourne"


def rate_of_change(series, window=7, polyorder=2):
    """Smoothed time derivative of a regularly spaced series, per hour.

    A Savitzky-Golay filter fits a low-degree polynomial to each moving
    window of readings and differentiates it (Week 7.1.11). Differentiating
    raw readings would amplify sensor noise; the window trades noise
    suppression against blurring of sharp events.

    Parameters
    ----------
    series : pandas.Series
        Gap-free, regularly spaced series with a DatetimeIndex
        (e.g. the reconstructed data).
    window : int, default 7
        Number of readings in each window (odd; 7 readings = 70 minutes).
    polyorder : int, default 2
        Degree of the local polynomial (must be less than ``window``).

    Returns
    -------
    pandas.Series
        Rate of change in units of the series per hour
        (e.g. degC/h for temperature, ppm/h for CO2).

    Raises
    ------
    ValueError
        If the series contains missing values.
    """
    if series.isna().any():
        raise ValueError("series contains NaN; reconstruct the gaps first")
    step_hours = (series.index[1] - series.index[0]).total_seconds() / 3600.0
    rate = savgol_filter(series.to_numpy(dtype=float), window, polyorder,
                         deriv=1, delta=step_hours)
    return pd.Series(rate, index=series.index, name=f"d_{series.name}_dt")


def daily_event(rate, start_hour, end_hour, kind="max", min_rate=0.0, tz=LOCAL_TZ):
    """Time of the strongest rise (or fall) within a daily search window.

    Parameters
    ----------
    rate : pandas.Series
        Output of :func:`rate_of_change`.
    start_hour, end_hour : float
        Local-time search window in hours. ``end_hour`` may exceed 24 to
        search past midnight (e.g. 20 to 28 means 8 pm to 4 am the next
        morning; the event is credited to the evening's date).
    kind : {"max", "min"}, default "max"
        "max" finds the fastest rise, "min" the fastest fall.
    min_rate : float, default 0.0
        Events weaker than this (absolute rate) are treated as "no event"
        and given a missing time, so quiet days are not forced to have one.
    tz : str, default "Australia/Melbourne"
        Local time zone (the building's clock, including daylight saving).

    Returns
    -------
    pandas.DataFrame
        One row per local date with columns ``time`` (local timestamp of the
        event, or NaT), ``hour`` (event time as decimal hours, e.g. 7.5 =
        07:30; values above 24 are after midnight), ``rate`` (the rate at
        that time) and ``weekday`` (True Monday-Friday).
    """
    if kind not in ("max", "min"):
        raise ValueError("kind must be 'max' or 'min'")
    local = rate.tz_convert(tz)
    hour = np.asarray(local.index.hour + local.index.minute / 60.0, dtype=float)
    day = local.index.tz_localize(None).normalize()
    if end_hour > 24:
        # Early-morning readings belong to the previous evening's window.
        early = hour < end_hour - 24
        hour = np.where(early, hour + 24, hour)
        day = day - pd.to_timedelta(early.astype(int), unit="D")
    inside = (hour >= start_hour) & (hour < end_hour)
    table = pd.DataFrame({"rate": local.to_numpy(), "time": local.index,
                          "hour": hour, "day": day})[inside]

    rows = []
    for date, group in table.groupby("day"):
        i = group["rate"].idxmax() if kind == "max" else group["rate"].idxmin()
        value = group.loc[i, "rate"]
        found = abs(value) >= min_rate
        rows.append((date,
                     group.loc[i, "time"] if found else pd.NaT,
                     group.loc[i, "hour"] if found else np.nan,
                     value))
    out = pd.DataFrame(rows, columns=["date", "time", "hour", "rate"]).set_index("date")
    out["weekday"] = out.index.dayofweek < 5
    return out


def time_features(index, tz=LOCAL_TZ):
    """Hour-of-day and weekend features for the decision-tree check (Week 5.2).

    Parameters
    ----------
    index : pandas.DatetimeIndex
        Time-zone-aware timestamps.
    tz : str, default "Australia/Melbourne"
        Local time zone.

    Returns
    -------
    pandas.DataFrame
        Columns ``hour`` (decimal local hour, 0-24) and ``weekend``
        (1 on Saturday/Sunday, else 0), with the original index.
    """
    local = index.tz_convert(tz)
    return pd.DataFrame({
        "hour": local.hour + local.minute / 60.0,
        "weekend": (local.dayofweek >= 5).astype(int),
    }, index=index)


def daily_profile(series, tz=LOCAL_TZ):
    """Average day: mean value at each time of day, weekdays vs weekends.

    Parameters
    ----------
    series : pandas.Series
        Regularly spaced series with a time-zone-aware DatetimeIndex.
    tz : str, default "Australia/Melbourne"
        Local time zone.

    Returns
    -------
    pandas.DataFrame
        Index = decimal local hour (0, 1/6, ..., 23 5/6); columns
        ``weekday`` and ``weekend`` holding the mean value at that time.
    """
    f = time_features(series.index, tz)
    table = pd.DataFrame({"value": series.to_numpy(), "hour": f["hour"].round(4),
                          "day_type": np.where(f["weekend"] == 1, "weekend", "weekday")})
    return table.pivot_table(index="hour", columns="day_type", values="value",
                             aggfunc="mean")