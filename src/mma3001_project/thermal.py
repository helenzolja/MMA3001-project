"""First-order (lumped) models of warm-up and cool-down (Week 7.2).

After the HVAC switches on, the room temperature rises towards a set-point;
after it switches off, it falls towards an unheated level. Both follow the
first-order ODE

    dT/dt = (T_inf - T) / tau,      T(0) = T0,

with exact solution  T(t) = T_inf + (T0 - T_inf) exp(-t / tau).

``T_inf`` is the level approached and ``tau`` [h] is the time constant:
after one ``tau`` the remaining distance to ``T_inf`` has shrunk to 37 %.
This module cuts the warm-up and cool-down segments out of the data, fits
``T_inf`` and ``tau`` by least squares (Week 7.2.15), validates them on
unseen days, and converts them into the times the schedule needs.

Example
-------
>>> from mma3001_project.thermal import extract_segments, fit_segments
>>> warm = extract_segments(recon["temp_c"], events["hvac_on_h"], length_h=2, pick="min")
>>> fits = fit_segments(warm, guess=(23.5, 0.5))
"""

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

LOCAL_TZ = "Australia/Melbourne"


def first_order_response(t, y_inf, tau, y0):
    """Exact solution of dy/dt = (y_inf - y)/tau with y(0) = y0.

    Parameters
    ----------
    t : float or array_like
        Time since the start of the segment [h].
    y_inf : float
        Level approached as t grows (set-point or unheated level).
    tau : float
        Time constant [h] (must be positive).
    y0 : float
        Value at t = 0.

    Returns
    -------
    numpy.ndarray
        y(t).
    """
    return y_inf + (y0 - y_inf) * np.exp(-np.asarray(t, dtype=float) / tau)


def time_to_reach(y0, target, y_inf, tau):
    """Time [h] for the first-order response to go from y0 to target.

    Solving target = y_inf + (y0 - y_inf) exp(-t/tau) for t gives
    t = tau * ln((y0 - y_inf) / (target - y_inf)).

    Parameters
    ----------
    y0, target, y_inf, tau : float
        Start value, target value, level approached and time constant [h].
        ``target`` must lie between ``y0`` and ``y_inf``.

    Returns
    -------
    float
        Time in hours.

    Raises
    ------
    ValueError
        If the target is never reached (not between y0 and y_inf).
    """
    ratio = (y0 - y_inf) / (target - y_inf)
    if not ratio >= 1:
        raise ValueError("target must lie between y0 and y_inf")
    return float(tau * np.log(ratio))


def extract_segments(series, event_hours, length_h, pick="min", before_h=1.0, tz=LOCAL_TZ):
    """Cut one response segment per day, starting where the response begins.

    The event times from :mod:`mma3001_project.patterns` mark the steepest
    point of each rise or fall. The segment starts at the last minimum
    (``pick="min"``, for a warm-up) or maximum (``pick="max"``, for a
    cool-down) in the ``before_h`` hours before that point, i.e. where the
    change began.

    Parameters
    ----------
    series : pandas.Series
        Gap-free series with a time-zone-aware DatetimeIndex.
    event_hours : pandas.Series
        Decimal local event hour per date (index = date), e.g. the
        ``hvac_on_h`` column of ``results/daily_events.csv``. NaN = no event.
    length_h : float
        Segment length [h].
    pick : {"min", "max"}, default "min"
        Start at the last minimum (rise) or last maximum (fall).
    before_h : float, default 1.0
        How far before the event to look for the start [h].
    tz : str, default "Australia/Melbourne"
        Local time zone.

    Returns
    -------
    list of tuple
        ``(date, t_hours, values)`` per day: time since the segment start
        [h] and the readings, both numpy arrays.
    """
    local = series.tz_convert(tz)
    out = []
    for date, hour in event_hours.dropna().items():
        event = pd.Timestamp(date).tz_localize(tz) + pd.Timedelta(hours=float(hour))
        before = local[event - pd.Timedelta(hours=before_h):event]
        if before.empty:
            continue
        reversed_ = before[::-1]                    # so ties pick the LAST extreme
        start = reversed_.idxmin() if pick == "min" else reversed_.idxmax()
        seg = local[start:start + pd.Timedelta(hours=length_h)]
        t = ((seg.index - start).total_seconds() / 3600.0).to_numpy()
        out.append((pd.Timestamp(date), t, seg.to_numpy(dtype=float)))
    return out


def fit_first_order(t, y, guess, tau_bounds=(0.05, 24.0)):
    """Fit y_inf and tau to one segment by least squares (Week 7.2.15).

    The starting value y0 is fixed to the first reading, so only the level
    approached (``y_inf``) and the time constant (``tau``) are fitted.

    Parameters
    ----------
    t, y : array_like
        Time since segment start [h] and readings.
    guess : tuple of float
        Starting guess (y_inf, tau).
    tau_bounds : tuple of float, default (0.05, 24.0)
        Allowed range for tau [h]. A fit that ends on a bound means the
        segment does not contain enough information to pin tau down.

    Returns
    -------
    dict
        ``y_inf``, ``tau`` [h], ``rmse`` (fit residual) and ``at_bound``
        (True if tau ended on one of its bounds).
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    result = least_squares(
        lambda p: first_order_response(t, p[0], p[1], y[0]) - y,
        x0=guess,
        bounds=([-np.inf, tau_bounds[0]], [np.inf, tau_bounds[1]]),
    )
    y_inf, tau = result.x
    return {
        "y_inf": float(y_inf),
        "tau": float(tau),
        "rmse": float(np.sqrt(np.mean(result.fun ** 2))),
        "at_bound": bool(np.isclose(tau, tau_bounds[0]) or np.isclose(tau, tau_bounds[1])),
    }


def fit_segments(segments, guess, tau_bounds=(0.05, 24.0)):
    """Fit :func:`fit_first_order` to every segment.

    Parameters
    ----------
    segments : list of tuple
        Output of :func:`extract_segments`.
    guess, tau_bounds
        Passed to :func:`fit_first_order`.

    Returns
    -------
    pandas.DataFrame
        One row per date with ``y0``, ``y_inf``, ``tau``, ``rmse``, ``at_bound``.
    """
    rows = []
    for date, t, y in segments:
        fit = fit_first_order(t, y, guess, tau_bounds)
        rows.append({"date": date, "y0": y[0], **fit})
    return pd.DataFrame(rows).set_index("date")


def prediction_errors(segments, y_inf, tau):
    """Errors when FIXED parameters predict each segment from its first reading.

    Validates fitted parameters on days not used for fitting (Week 7.2.15).
    Also returns the errors of a no-change baseline that predicts the first
    reading for the whole segment.

    Parameters
    ----------
    segments : list of tuple
        Output of :func:`extract_segments`.
    y_inf, tau : float
        Fixed model parameters.

    Returns
    -------
    tuple of numpy.ndarray
        (model errors, no-change errors), each prediction - reading.
    """
    model, baseline = [], []
    for _, t, y in segments:
        model.append(first_order_response(t, y_inf, tau, y[0]) - y)
        baseline.append(y[0] - y)
    return np.concatenate(model), np.concatenate(baseline)