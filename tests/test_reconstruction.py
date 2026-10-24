"""Tests for mma3001_project.reconstruction (run with: python -m pytest)."""
import numpy as np
import pandas as pd
import pytest

from mma3001_project.reconstruction import (
    METHODS,
    bootstrap_rmse,
    evaluate_methods,
    fill_gaps,
    reconstruct_frame,
    sample_gap_starts,
    summarise_errors,
)


def make_series(values):
    """Values on a 10-minute grid, like the cleaned sensor data."""
    index = pd.date_range("2024-03-01", periods=len(values), freq="10min", tz="UTC")
    return pd.Series(values, index=index, dtype=float, name="co2_ppm")


@pytest.mark.parametrize("method", METHODS)
def test_observed_values_unchanged_and_edges_not_extrapolated(method):
    s = make_series([np.nan, 400, 410, np.nan, 430, 440, np.nan, np.nan])
    filled = fill_gaps(s, method=method)
    observed = s.notna()
    assert np.array_equal(filled[observed], s[observed])   # passes through the data
    assert np.isfinite(filled.iloc[3])                      # interior gap filled
    assert filled.iloc[[0, 6, 7]].isna().all()              # edge gaps left empty


@pytest.mark.parametrize("method", ["linear", "cubic", "pchip"])
def test_straight_line_recovered_exactly(method):
    truth = 400 + 5 * np.arange(12.0)
    s = make_series(truth)
    s.iloc[[4, 5, 6]] = np.nan
    assert np.allclose(fill_gaps(s, method=method), truth)


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


def test_gap_starts_only_where_truth_exists():
    values = 400 + np.arange(100.0)
    values[[10, 11, 50]] = np.nan                       # genuinely missing
    s = make_series(values)
    starts = sample_gap_starts(s, 3, 30, np.random.default_rng(0))
    ok = s.notna().to_numpy()
    for st in starts:
        assert ok[st - 1:st + 4].all()                  # neighbours and hidden slots real


def test_validation_gives_zero_error_for_linear_on_a_line():
    s = make_series(400 + 2 * np.arange(300.0))
    table = summarise_errors(evaluate_methods(s, [1, 2], n_gaps=10, methods=("linear", "locf")))
    assert float(table.loc[("linear", 2), "RMSE"]) < 1e-9
    assert float(table.loc[("locf", 2), "RMSE"]) > 1    # LOCF lags a trend


def test_bootstrap_interval_contains_estimate():
    errors = pd.DataFrame({"gap_id": np.repeat(np.arange(50), 3),
                           "error": np.random.default_rng(1).normal(0, 10, 150)})
    rmse, lo, hi = bootstrap_rmse(errors, n_boot=500)
    assert lo <= rmse <= hi