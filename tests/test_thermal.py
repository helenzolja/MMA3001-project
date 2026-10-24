"""Tests for mma3001_project.thermal (run with: python -m pytest)."""
import numpy as np
import pandas as pd
import pytest

from mma3001_project.thermal import (
    extract_segments,
    first_order_response,
    fit_first_order,
    fit_segments,
    prediction_errors,
    time_to_reach,
)

Y_INF, TAU, Y0 = 24.0, 0.75, 21.0


def test_response_starts_at_y0_and_satisfies_the_ode():
    t = np.linspace(0, 3, 3001)
    y = first_order_response(t, Y_INF, TAU, Y0)
    assert y[0] == pytest.approx(Y0)
    dydt = np.gradient(y, t)                       # numerical derivative (Week 7.1)
    assert np.allclose(dydt[1:-1], ((Y_INF - y) / TAU)[1:-1], atol=1e-4)


def test_time_to_reach_inverts_the_response():
    t = time_to_reach(Y0, 23.0, Y_INF, TAU)
    assert first_order_response(t, Y_INF, TAU, Y0) == pytest.approx(23.0)
    with pytest.raises(ValueError):
        time_to_reach(Y0, 25.0, Y_INF, TAU)        # beyond the set-point: never reached


def test_fit_recovers_known_parameters():
    rng = np.random.default_rng(0)
    t = np.arange(0, 2.01, 1 / 6)                  # every 10 minutes for 2 hours
    y = first_order_response(t, Y_INF, TAU, Y0) + rng.normal(0, 0.05, t.size)
    y[0] = Y0
    fit = fit_first_order(t, y, guess=(23.0, 0.3))
    assert fit["y_inf"] == pytest.approx(Y_INF, abs=0.2)
    assert fit["tau"] == pytest.approx(TAU, abs=0.1)
    assert not fit["at_bound"]


def test_extract_segments_starts_at_last_minimum():
    index = pd.date_range("2024-05-06 06:00", periods=24, freq="10min",
                          tz="Australia/Melbourne").tz_convert("UTC")
    values = np.r_[np.full(8, 20.0), 20 + 4 * (1 - np.exp(-np.arange(16) / 4.5))]
    series = pd.Series(values, index=index)
    events = pd.Series([7.5], index=[pd.Timestamp("2024-05-06")])   # steepest at 07:30
    (date, t, y), = extract_segments(series, events, length_h=2, pick="min")
    assert t[0] == 0 and y[0] == 20.0
    assert len(y) == 13                             # 07:20 to 09:20 inclusive


def test_prediction_errors_zero_for_exact_data():
    t = np.arange(0, 2.01, 1 / 6)
    segs = [(pd.Timestamp("2024-06-03"), t, first_order_response(t, Y_INF, TAU, Y0))]
    model, baseline = prediction_errors(segs, Y_INF, TAU)
    assert np.allclose(model, 0)
    assert np.abs(baseline).max() > 2               # "no change" misses the warm-up
    assert len(fit_segments(segs, (23.0, 0.5))) == 1
