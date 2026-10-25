"""Proposed HVAC schedule: simulate room temperature and measure savings.

Uses the first-order warm-up and cool-down models from
:mod:`mma3001_project.thermal` to simulate the room temperature under any
on/off schedule, checks whether occupied times stay comfortable, and
measures run-hours and heated degree-hours (Week 6 trapezoidal integration)
as indicators of energy use.

Example
-------
>>> on = schedule_mask(temp.index, weekday=(8.0, 19.33))      # weekends off
>>> T = simulate_temperature(on, temp.iloc[0], warm=(24.2, 0.77), cool=(21.0, 3.69))
"""

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid

LOCAL_TZ = "Australia/Melbourne"
STEP_H = 10 / 60          # the data's 10-minute spacing, in hours


def _local_hour_and_day(index, tz=LOCAL_TZ):
    """Decimal local hour (0-24), local date and weekday flag for each timestamp."""
    local = index.tz_convert(tz)
    hour = np.asarray(local.hour + local.minute / 60.0, dtype=float)
    return hour, local.tz_localize(None).normalize(), np.asarray(local.dayofweek < 5)


def schedule_mask(index, weekday, weekend=None, tz=LOCAL_TZ):
    """True where the HVAC is on under a daily on/off schedule.

    Parameters
    ----------
    index : pandas.DatetimeIndex
        Time-zone-aware timestamps (the data grid).
    weekday : tuple of float
        (on_hour, off_hour) in local decimal hours for Monday-Friday,
        e.g. (7.0, 24.0) = 07:00 to midnight.
    weekend : tuple of float or None, default None
        Same for Saturday-Sunday; None means off all weekend.
    tz : str, default "Australia/Melbourne"
        Local time zone of the timer.

    Returns
    -------
    numpy.ndarray of bool
        One entry per timestamp.
    """
    hour, _, is_weekday = _local_hour_and_day(index, tz)
    on = is_weekday & (hour >= weekday[0]) & (hour < weekday[1])
    if weekend is not None:
        on |= ~is_weekday & (hour >= weekend[0]) & (hour < weekend[1])
    return on


def simulate_temperature(on, T0, warm, cool, step_h=STEP_H):
    """Room temperature under an on/off schedule, from the first-order models.

    Over each step the temperature follows the exact first-order solution
    towards the warm-up level (HVAC on) or the unheated level (HVAC off):
    T[n+1] = T_inf + (T[n] - T_inf) * exp(-step_h / tau). Because
    0 < exp(-step_h/tau) < 1 for any step, this update cannot become
    unstable, unlike explicit Euler (Week 7.3).

    Parameters
    ----------
    on : array_like of bool
        HVAC state for each step (e.g. from :func:`schedule_mask`).
    T0 : float
        Starting temperature [degC].
    warm, cool : tuple of float
        (T_inf, tau [h]) for HVAC on and HVAC off.
    step_h : float, default 1/6
        Time step [h].

    Returns
    -------
    numpy.ndarray
        Simulated temperature at every step [degC].
    """
    on = np.asarray(on, dtype=bool)
    T = np.empty(len(on))
    T[0] = T0
    decay_on = np.exp(-step_h / warm[1])
    decay_off = np.exp(-step_h / cool[1])
    for n in range(len(on) - 1):
        T_inf, decay = (warm[0], decay_on) if on[n] else (cool[0], decay_off)
        T[n + 1] = T_inf + (T[n] - T_inf) * decay
    return T


def arrival_onset(co2, rise_ppm=30.0, base_hours=(5, 7), search_hours=(6, 14), tz=LOCAL_TZ):
    """First time each day that CO2 rises clearly above the early-morning level.

    Parameters
    ----------
    co2 : pandas.Series
        Gap-free CO2 [ppm] with a time-zone-aware DatetimeIndex.
    rise_ppm : float, default 30
        Rise above the baseline that counts as "people present"
        (about 4 times the reading-to-reading noise after smoothing).
    base_hours : tuple of float, default (5, 7)
        Local hours whose median CO2 is the empty-room baseline.
    search_hours : tuple of float, default (6, 14)
        Local hours searched for the first rise.
    tz : str, default "Australia/Melbourne"
        Local time zone.

    Returns
    -------
    pandas.Series
        Onset as decimal local hour per date (NaN if no clear rise).
    """
    smooth = co2.tz_convert(tz).rolling(3, center=True, min_periods=1).mean()
    hour, day, _ = _local_hour_and_day(co2.index, tz)
    table = pd.DataFrame({"co2": smooth.to_numpy(), "hour": hour, "day": day})
    onsets = {}
    for date, g in table.groupby("day"):
        base = g.loc[(g["hour"] >= base_hours[0]) & (g["hour"] < base_hours[1]), "co2"].median()
        window = g[(g["hour"] >= search_hours[0]) & (g["hour"] < search_hours[1])]
        above = window.loc[window["co2"] - base > rise_ppm, "hour"]
        onsets[date] = above.iloc[0] if len(above) else np.nan
    return pd.Series(onsets, name="onset_h")


def occupied_mask(index, start_hours, end_hours, tz=LOCAL_TZ):
    """True for timestamps inside each day's occupied period.

    Parameters
    ----------
    index : pandas.DatetimeIndex
        Time-zone-aware timestamps.
    start_hours, end_hours : pandas.Series
        Per-date start and end of occupancy (decimal local hours); dates
        missing from either, or NaN, count as unoccupied.
    tz : str, default "Australia/Melbourne"
        Local time zone.

    Returns
    -------
    numpy.ndarray of bool
        One entry per timestamp; True where start <= local hour < end.
    """
    hour, day, _ = _local_hour_and_day(index, tz)
    start = start_hours.reindex(day).to_numpy()
    end = end_hours.reindex(day).to_numpy()
    with np.errstate(invalid="ignore"):          # NaN comparisons are simply False
        return (hour >= start) & (hour < end)


def schedule_metrics(on, T, occupied, comfort, unheated, step_h=STEP_H):
    """Energy indicators and comfort check for one simulated schedule.

    Parameters
    ----------
    on : array_like of bool
        HVAC state per step.
    T : array_like
        Simulated temperature per step [degC].
    occupied : array_like of bool
        Occupied steps.
    comfort : float
        Lowest acceptable occupied temperature [degC].
    unheated : float
        Temperature the room settles to with the HVAC off [degC].
    step_h : float, default 1/6
        Time step [h].

    Returns
    -------
    dict
        ``run_h_per_week``: hours the HVAC runs per week;
        ``degree_h_per_week``: integral of (T - unheated) over time per week,
        by the trapezoidal rule [degC h], an indicator of heat supplied;
        ``occupied_below_comfort_pct``: % of occupied steps below ``comfort``.
    """
    on, T, occupied = np.asarray(on, bool), np.asarray(T, float), np.asarray(occupied, bool)
    weeks = len(T) * step_h / (24 * 7)                 # normalise so periods compare
    t = np.arange(len(T)) * step_h
    excess = np.maximum(T - unheated, 0.0)             # only heat held above free-running level
    return {
        "run_h_per_week": on.sum() * step_h / weeks,
        "degree_h_per_week": trapezoid(excess, t) / weeks,
        "occupied_below_comfort_pct": 100 * np.mean(T[occupied] < comfort) if occupied.any() else np.nan,
    }