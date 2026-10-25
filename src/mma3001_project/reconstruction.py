"""Fill missing sensor readings by interpolation and validate the choice of method.

The cleaned series from :mod:`mma3001_project.load` sits on a regular
10-minute grid, where an empty slot is ``NaN``. This module fills those slots
from the observed readings on either side (Week 3 interpolation) and never
extrapolates: a gap at the very start or end of the record stays ``NaN``.

Methods compared:

* ``"locf"``   - last observation carried forward (baseline)
* ``"linear"`` - piecewise-linear interpolation (3.1 / 3.4)
* ``"cubic"``  - cubic spline (3.4)
* ``"pchip"``  - shape-preserving piecewise cubic Hermite (3.4)

The method is chosen by hiding real readings, filling them and measuring the
error (:func:`evaluate_methods`).

Example
-------
>>> from mma3001_project.reconstruction import reconstruct_frame
>>> recon = reconstruct_frame(clean, method="linear")
"""

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, PchipInterpolator, make_interp_spline

METHODS = ("locf", "linear", "cubic", "pchip")


def fill_gaps(series, method="linear"):
    """Fill the interior gaps of a regularly spaced series.

    Parameters
    ----------
    series : pandas.Series
        Series with a DatetimeIndex; ``NaN`` marks a missing reading.
    method : {"locf", "linear", "cubic", "pchip"}, default "linear"
        Reconstruction method (see the module docstring).

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

    # Hours since the first slot: a numeric x-axis for the interpolants.
    t = ((series.index - series.index[0]).total_seconds() / 3600.0).to_numpy()
    t_obs, y_obs = t[observed], y[observed]
    missing = ~observed
    filled = y.copy()

    if method == "locf":
        filled = series.ffill().to_numpy(dtype=float, copy=True)
    elif method == "linear":
        filled[missing] = make_interp_spline(t_obs, y_obs, k=1)(t[missing])
    elif method == "cubic":
        filled[missing] = CubicSpline(t_obs, y_obs)(t[missing])
    elif method == "pchip":
        filled[missing] = PchipInterpolator(t_obs, y_obs)(t[missing])

    # Interpolation only: blank anything outside the observed time range.
    outside = (t < t_obs[0]) | (t > t_obs[-1])
    filled[missing & outside] = np.nan
    return pd.Series(filled, index=series.index, name=series.name)


def reconstruct_frame(frame, method="linear"):
    """Fill every column of a cleaned frame and flag what was reconstructed.

    Parameters
    ----------
    frame : pandas.DataFrame
        Output of :func:`mma3001_project.load.clean_sensor_frame`.
    method : str, default "linear"
        Passed to :func:`fill_gaps`.

    Returns
    -------
    pandas.DataFrame
        The filled columns plus a boolean ``<column>_reconstructed`` column
        for each, True where the value was filled rather than measured.
    """
    out = pd.DataFrame(index=frame.index)
    for col in frame.columns:
        filled = fill_gaps(frame[col], method=method)
        out[col] = filled
        out[f"{col}_reconstructed"] = frame[col].isna() & filled.notna()
    return out


def sample_gap_starts(series, length, n, rng):
    """Choose random positions for artificial gaps of ``length`` slots.

    A start ``s`` is allowed only if every hidden slot holds a real reading
    (so there is a true value to score against) and the slots just before
    (``s-1``) and after (``s+length``) are observed, like a real gap.

    Parameters
    ----------
    series : pandas.Series
        Regularly spaced series; ``NaN`` marks a genuinely missing reading.
    length : int
        Gap length in slots (1 slot = 10 minutes).
    n : int
        Number of gaps to choose.
    rng : numpy.random.Generator
        Seeded random generator, so results are reproducible.

    Returns
    -------
    numpy.ndarray
        Integer start positions (without repeats when enough are available).

    Raises
    ------
    ValueError
        If no allowed start position exists.
    """
    ok = series.notna().to_numpy()
    counts = np.concatenate(([0], np.cumsum(ok)))     # running count of readings
    s = np.arange(1, len(ok) - length)                # needs s-1 and s+length
    full = (counts[s + length] - counts[s]) == length # every hidden slot observed
    starts = s[full & ok[s - 1] & ok[s + length]]
    if len(starts) == 0:
        raise ValueError(f"no allowed start positions for a gap of {length} slots")
    return rng.choice(starts, size=n, replace=len(starts) < n)


def evaluate_methods(series, lengths, n_gaps=200, seed=0, methods=METHODS, window=72):
    """Hide real readings, fill them with each method, record the errors.

    For every gap length, ``n_gaps`` artificial gaps are placed at random.
    Each method fills the gap from the surrounding ``window`` slots on each
    side, and the error (filled - true) is recorded at every hidden slot.

    Parameters
    ----------
    series : pandas.Series
        Regularly spaced series with a DatetimeIndex.
    lengths : list of int
        Gap lengths to test, in slots (1 slot = 10 minutes).
    n_gaps : int, default 200
        Number of artificial gaps per length.
    seed : int, default 0
        Random seed (fixed, so the experiment is reproducible).
    methods : sequence of str, default METHODS
        Methods passed to :func:`fill_gaps`.
    window : int, default 72
        Slots of context kept on each side of a gap (72 = 12 hours).

    Returns
    -------
    pandas.DataFrame
        One row per (method, hidden reading) with columns ``method``,
        ``length``, ``gap_id`` and ``error``.
    """
    rng = np.random.default_rng(seed)
    values = series.to_numpy(dtype=float)
    rows = []
    gap_id = 0
    for length in lengths:
        for s in sample_gap_starts(series, length, n_gaps, rng):
            # Local window keeps each fill O(window), not O(series); see 07_performance.
            lo, hi = max(0, s - window), min(len(series), s + length + window)
            local = series.iloc[lo:hi].copy()
            local.iloc[s - lo:s - lo + length] = np.nan       # hide the readings
            truth = values[s:s + length]
            for m in methods:
                pred = fill_gaps(local, method=m).to_numpy()[s - lo:s - lo + length]
                rows += [(m, length, gap_id, e) for e in pred - truth]
            gap_id += 1
    return pd.DataFrame(rows, columns=["method", "length", "gap_id", "error"])


def summarise_errors(errors, by=("method", "length")):
    """Pooled RMSE, MAE and bias for each group of an :func:`evaluate_methods` table.

    Parameters
    ----------
    errors : pandas.DataFrame
        Output of :func:`evaluate_methods`.
    by : sequence of str, default ("method", "length")
        Columns to group by.

    Returns
    -------
    pandas.DataFrame
        Columns ``RMSE``, ``MAE``, ``bias`` (same units as the data) and
        ``n`` for each group.
    """
    g = errors.groupby(list(by))["error"]
    return pd.DataFrame({
        "RMSE": np.sqrt(g.apply(lambda e: np.mean(e ** 2))),
        "MAE": g.apply(lambda e: np.mean(np.abs(e))),
        "bias": g.mean(),
        "n": g.size(),
    })


def bootstrap_rmse(errors, n_boot=2000, seed=0, level=0.95):
    """Bootstrap confidence interval for a pooled RMSE, resampling whole gaps.

    Readings inside one gap are not independent, so whole gaps (not single
    readings) are resampled with replacement (Week 5.5.11).

    Parameters
    ----------
    errors : pandas.DataFrame
        Rows of :func:`evaluate_methods` for ONE method.
    n_boot : int, default 2000
        Number of bootstrap resamples.
    seed : int, default 0
        Random seed.
    level : float, default 0.95
        Confidence level.

    Returns
    -------
    tuple of float
        (RMSE, lower bound, upper bound).
    """
    # Reduce each gap to (sum of squared errors, count) so a resample is a cheap sum.
    per_gap = errors.groupby("gap_id")["error"].agg(
        sse=lambda e: np.sum(e ** 2), n="size")
    sse, n = per_gap["sse"].to_numpy(), per_gap["n"].to_numpy()
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(sse), size=(n_boot, len(sse)))   # gap indices per resample
    boot = np.sqrt(sse[pick].sum(axis=1) / n[pick].sum(axis=1))
    alpha = (1 - level) / 2
    return (float(np.sqrt(sse.sum() / n.sum())), float(np.quantile(boot, alpha)), float(np.quantile(boot, 1 - alpha)))