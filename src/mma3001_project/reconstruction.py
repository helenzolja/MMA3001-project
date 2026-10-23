"""Reconstruction of missing sensor readings using interpolation methods.

The cleaned series from :mod:`mma3001_project.load` sits on a regular
10-minute grid, where an empty slot is ``NaN``. This module fills those slots
using only the observed readings on either side of each gap (interpolation).
It never extrapolates: a gap at the very start or end of the record is left
as ``NaN``.

Methods (unit section in brackets):

* ``"locf"``   - last observation carried forward (baseline)
* ``"linear"`` - piecewise-linear interpolation (3.1 / 3.4)
* ``"cubic"``  - cubic spline, not-a-knot or natural end conditions (3.4)
* ``"pchip"``  - shape-preserving piecewise cubic Hermite (3.4)
* ``"newton"`` - local Newton divided-difference polynomial (3.3)

Example
-------
>>> from mma3001_project.reconstruction import fill_gaps
>>> filled = fill_gaps(clean["co2_ppm"], method="linear")
"""

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, PchipInterpolator, make_interp_spline

"""listing the 5 different methods used"""
METHODS = ("locf", "linear", "cubic", "pchip", "newton")

"""turning timestamps into hourse in order to create a numeric x axis"""
def _hours(index):
    """Convert a DatetimeIndex to hours since its first entry (a numeric x-axis)."""
    return ((index - index[0]).total_seconds() / 3600.0).to_numpy()

"""marking missing slots as True and reporting gap lengths"""
def find_gaps(series):
    """List every run of missing values in a regularly spaced series.

    Parameters
    ----------
    series : pandas.Series
        Series on a regular grid; ``NaN`` marks a missing reading.

    Returns
    -------
    pandas.DataFrame
        One row per gap with columns ``start`` and ``end`` (timestamps of the
        first and last missing slot), ``n_missing`` (gap length in slots) and
        ``interior`` (True if there is an observed reading on both sides, so
        the gap can be interpolated rather than extrapolated).
    """
    missing = series.isna().to_numpy()
    padded = np.concatenate(([False], missing, [False]))
    edges = np.diff(padded.astype(int))
    starts = np.flatnonzero(edges == 1)        # first missing slot of each run
    ends = np.flatnonzero(edges == -1) - 1     # last missing slot of each run
    n = len(series)
    return pd.DataFrame({
        "start": series.index[starts],
        "end": series.index[ends],
        "n_missing": ends - starts + 1,
        "interior": (starts > 0) & (ends < n - 1),
    })

"""newton implementation"""
def divided_differences(x, y):
    """Newton divided-difference coefficients (top row of the table).

    Parameters
    ----------
    x, y : array_like
        Distinct nodes and the values at those nodes.

    Returns
    -------
    numpy.ndarray
        Coefficients ``c`` such that
        ``p(t) = c0 + c1 (t-x0) + c2 (t-x0)(t-x1) + ...``.
    """
    x = np.asarray(x, dtype=float)
    c = np.asarray(y, dtype=float).copy()
    for j in range(1, len(x)):
        # Column j of the table, written over the entries no longer needed.
        c[j:] = (c[j:] - c[j - 1:-1]) / (x[j:] - x[:-j])
    return c


def newton_evaluate(coefficients, x_nodes, t):
    """Evaluate a Newton-form polynomial at ``t`` using nested (Horner) form.

    Parameters
    ----------
    coefficients : array_like
        Output of :func:`divided_differences`.
    x_nodes : array_like
        The nodes used to build the coefficients.
    t : float or array_like
        Point(s) at which to evaluate.

    Returns
    -------
    float or numpy.ndarray
        Polynomial value(s) at ``t``.
    """
    c = np.asarray(coefficients, dtype=float)
    x_nodes = np.asarray(x_nodes, dtype=float)
    p = c[-1] * np.ones_like(np.asarray(t, dtype=float))
    for k in range(len(c) - 2, -1, -1):
        p = p * (t - x_nodes[k]) + c[k]
    return p

"""building newton polynomial from 2 before and 2 after readings from the gap"""
def _newton_fill(t_obs, y_obs, t_missing, n_points):
    """Fill each missing time with a Newton polynomial through nearby readings.

    The ``n_points`` nearest observed readings (half before, half after the
    missing time) are used, so the polynomial stays local and low degree.
    """
    half = n_points // 2
    out = np.empty(len(t_missing))
    positions = np.searchsorted(t_obs, t_missing)
    for i, (t, pos) in enumerate(zip(t_missing, positions)):
        lo = max(0, pos - half)
        hi = min(len(t_obs), lo + n_points)
        lo = max(0, hi - n_points)
        nodes = t_obs[lo:hi]
        out[i] = newton_evaluate(divided_differences(nodes, y_obs[lo:hi]), nodes, t)
    return out

"""fitting chosen method to measured points to evaluate missing times. curve passes through data
and avoid extrapolation by leaving gaps at the start and end of the series"""
def fill_gaps(series, method="linear", bc_type="not-a-knot", newton_points=4):
    """Fill the interior gaps of a regularly spaced series.

    Parameters
    ----------
    series : pandas.Series
        Series with a DatetimeIndex; ``NaN`` marks a missing reading.
    method : {"locf", "linear", "cubic", "pchip", "newton"}, default "linear"
        Reconstruction method (see the module docstring).
    bc_type : {"not-a-knot", "natural"}, default "not-a-knot"
        End condition for ``method="cubic"`` only.
    newton_points : int, default 4
        Number of nearby readings used by ``method="newton"``
        (4 points gives a local cubic).

    Returns
    -------
    pandas.Series
        Copy of ``series`` with interior gaps filled. Observed readings are
        unchanged, and gaps before the first or after the last observation
        stay ``NaN`` (no extrapolation).

    Raises
    ------
    ValueError
        If ``method`` is unknown or fewer than two readings are observed.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    y = series.to_numpy(dtype=float)
    observed = ~np.isnan(y)
    if observed.sum() < 2:
        raise ValueError("need at least two observed readings to interpolate")

    t = _hours(series.index)
    t_obs, y_obs = t[observed], y[observed]
    missing = ~observed
    filled = y.copy()

    if method == "locf":
        filled = series.ffill().to_numpy(dtype=float, copy=True)
    elif method == "linear":
        filled[missing] = make_interp_spline(t_obs, y_obs, k=1)(t[missing])
    elif method == "cubic":
        filled[missing] = CubicSpline(t_obs, y_obs, bc_type=bc_type)(t[missing])
    elif method == "pchip":
        filled[missing] = PchipInterpolator(t_obs, y_obs)(t[missing])
    elif method == "newton":
        filled[missing] = _newton_fill(t_obs, y_obs, t[missing], newton_points)

    # Interpolation only: blank anything outside the observed time range.
    outside = (t < t_obs[0]) | (t > t_obs[-1])
    filled[missing & outside] = np.nan
    return pd.Series(filled, index=series.index, name=series.name)

"""filling env variables and flagging what was reconstructed"""
def reconstruct_frame(frame, method="linear", **kwargs):
    """Fill every column of a cleaned frame and flag what was reconstructed.

    Parameters
    ----------
    frame : pandas.DataFrame
        Output of :func:`mma3001_project.load.clean_sensor_frame`.
    method : str, default "linear"
        Passed to :func:`fill_gaps`.
    **kwargs
        Extra options passed to :func:`fill_gaps`.

    Returns
    -------
    pandas.DataFrame
        The filled columns plus a boolean ``<column>_reconstructed`` column
        for each, True where the value was filled rather than measured.
    """
    out = pd.DataFrame(index=frame.index)
    for col in frame.columns:
        filled = fill_gaps(frame[col], method=method, **kwargs)
        out[col] = filled
        out[f"{col}_reconstructed"] = frame[col].isna() & filled.notna()
    return out

"""comparing reconstructed values to the hidden true readings and reporting RMSE, MAE, bias and n"""
def error_metrics(truth, prediction):
    """Error of reconstructed values against hidden true readings.

    Only positions where both values are finite are scored.

    Parameters
    ----------
    truth, prediction : array_like
        True (hidden) readings and reconstructed values.

    Returns
    -------
    dict
        ``RMSE``, ``MAE`` and ``bias`` (mean of prediction - truth; positive
        means the method over-estimates), in the units of the data, plus
        ``n`` (number of scored values).
    """
    truth = np.asarray(truth, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    ok = np.isfinite(truth) & np.isfinite(prediction)
    err = prediction[ok] - truth[ok]
    if err.size == 0:
        return {"RMSE": np.nan, "MAE": np.nan, "bias": np.nan, "n": 0}
    return {
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MAE": float(np.mean(np.abs(err))),
        "bias": float(np.mean(err)),
        "n": int(err.size),
    }