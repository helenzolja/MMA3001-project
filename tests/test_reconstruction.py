"""Tests for mma3001_project.reconstruction (run with: python -m pytest)."""
from mma3001_project.reconstruction import (
    METHODS,
    bootstrap_rmse,
    divided_differences,
    error_metrics,
    evaluate_methods,
    fill_gaps,
    find_gaps,
    is_occupied_hours,
    newton_evaluate,
    reconstruct_frame,
    sample_gap_starts,
    summarise_errors,
)
import numpy as np
import pandas as pd
import pytest
from scipy.interpolate import BarycentricInterpolator

from mma3001_project.reconstruction import (
    METHODS,
    divided_differences,
    error_metrics,
    fill_gaps,
    find_gaps,
    newton_evaluate,
    reconstruct_frame,
)


def make_series(values):
    """Values on a 10-minute grid, like the cleaned sensor data."""
    index = pd.date_range("2024-03-01", periods=len(values), freq="10min", tz="UTC")
    return pd.Series(values, index=index, dtype=float, name="co2_ppm")


def test_newton_reproduces_a_cubic_exactly():
    # A cubic through 4 nodes is unique, so Newton must recover it everywhere.
    x = np.array([0.0, 1.0, 2.5, 4.0])
    f = lambda t: 2 * t**3 - t + 5
    c = divided_differences(x, f(x))
    t = np.linspace(0, 4, 9)
    assert np.allclose(newton_evaluate(c, x, t), f(t))


def test_newton_matches_scipy_barycentric():
    # Cross-check our own code against an independent library implementation.
    x = np.array([0.0, 0.5, 1.7, 2.0, 3.1])
    y = np.array([450.0, 470.0, 462.0, 455.0, 480.0])
    t = np.linspace(0, 3.1, 20)
    ours = newton_evaluate(divided_differences(x, y), x, t)
    assert np.allclose(ours, BarycentricInterpolator(x, y)(t))


def test_find_gaps_lengths_and_edges():
    s = make_series([np.nan, 1, 2, np.nan, np.nan, 5, 6, np.nan])
    gaps = find_gaps(s)
    assert gaps["n_missing"].tolist() == [1, 2, 1]
    assert gaps["interior"].tolist() == [False, True, False]


@pytest.mark.parametrize("method", METHODS)
def test_observed_values_unchanged_and_edges_not_extrapolated(method):
    s = make_series([np.nan, 400, 410, np.nan, 430, 440, np.nan, np.nan])
    filled = fill_gaps(s, method=method)
    observed = s.notna()
    assert np.array_equal(filled[observed], s[observed])   # passes through the data
    assert np.isfinite(filled.iloc[3])                      # interior gap filled
    assert filled.iloc[[0, 6, 7]].isna().all()              # edge gaps left empty


@pytest.mark.parametrize("method", ["linear", "cubic", "pchip", "newton"])
def test_straight_line_recovered_exactly(method):
    truth = 400 + 5 * np.arange(12.0)
    s = make_series(truth)
    s.iloc[[4, 5, 6]] = np.nan
    assert np.allclose(fill_gaps(s, method=method), truth)


def test_pchip_does_not_overshoot_a_step():
    # A sudden rise (e.g. HVAC start-up): the cubic spline overshoots, PCHIP doesn't.
    # The missing reading sits just after the jump, on the flat part.
    s = make_series([400, 400, 400, 400, 600, np.nan, 600, 600, 600])
    pchip = fill_gaps(s, method="pchip")
    cubic = fill_gaps(s, method="cubic")
    assert pchip.iloc[5] == pytest.approx(600.0)    # stays flat
    assert cubic.iloc[5] > 620                       # overshoots (about 650)


def test_unknown_method_rejected():
    with pytest.raises(ValueError, match="method"):
        fill_gaps(make_series([1, np.nan, 3]), method="magic")


def test_reconstruct_frame_flags_filled_values():
    frame = pd.DataFrame({
        "co2_ppm": make_series([400, np.nan, 420, 430]),
        "temp_c": make_series([22, 22.5, np.nan, 23]),
    })
    out = reconstruct_frame(frame, method="linear")
    assert out["co2_ppm"].tolist() == [400, 410, 420, 430]
    assert out["co2_ppm_reconstructed"].tolist() == [False, True, False, False]
    assert out["temp_c_reconstructed"].sum() == 1


def test_error_metrics():
    m = error_metrics([400, 410, np.nan, 430], [402, 406, 500, 430])
    assert m["n"] == 3
    assert m["MAE"] == pytest.approx(2.0)
    assert m["bias"] == pytest.approx(-2 / 3)
    assert m["RMSE"] == pytest.approx(np.sqrt(20 / 3))



def test_gap_starts_have_real_readings_either_side():
    values = 400 + np.arange(100.0)
    values[[10, 11, 50]] = np.nan                      # genuinely missing
    s = make_series(values)
    rng = np.random.default_rng(0)
    starts = sample_gap_starts(s, 3, 30, rng)
    ok = s.notna().to_numpy()
    assert ok[starts - 1].all() and ok[starts + 3].all()


def test_gap_starts_reproducible_with_seed():
    s = make_series(400 + np.arange(100.0))
    a = sample_gap_starts(s, 5, 10, np.random.default_rng(42))
    b = sample_gap_starts(s, 5, 10, np.random.default_rng(42))
    assert np.array_equal(a, b)


def test_linear_has_zero_error_on_a_straight_line():
    s = make_series(400 + 2 * np.arange(300.0))
    errors = evaluate_methods(s, [1, 6], n_gaps=10, methods=("linear", "locf"))
    table = summarise_errors(errors)
    assert table.loc[("linear", 6), "RMSE"] < 1e-9
    assert table.loc[("locf", 6), "RMSE"] > 1                # LOCF lags a trend


def test_bootstrap_interval_contains_estimate():
    errors = pd.DataFrame({"gap_id": np.repeat(np.arange(50), 3),
                           "error": np.random.default_rng(1).normal(0, 10, 150)})
    rmse, lo, hi = bootstrap_rmse(errors, n_boot=500)
    assert lo <= rmse <= hi


def test_occupied_hours():
    # Monday 10:00 and Saturday 10:00 in Melbourne (UTC+10 in June)
    index = pd.DatetimeIndex(["2024-06-03 00:00", "2024-06-08 00:00"], tz="UTC")
    assert is_occupied_hours(index).tolist() == [True, False]