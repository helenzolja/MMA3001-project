"""Tests for mma3001_project.schedule (run with: python -m pytest)."""
import numpy as np
import pandas as pd
import pytest

from mma3001_project.schedule import (
    arrival_onset,
    occupied_mask,
    schedule_mask,
    schedule_metrics,
    simulate_temperature,
)
from mma3001_project.thermal import first_order_response

WARM, COOL = (24.0, 0.75), (21.0, 3.5)


def week_index():
    """One week every 10 minutes, starting Monday 00:00 Melbourne time."""
    return pd.date_range("2024-05-06", periods=7 * 144, freq="10min",
                         tz="Australia/Melbourne").tz_convert("UTC")


def test_schedule_mask_weekdays_only():
    on = schedule_mask(week_index(), weekday=(8.0, 19.0))
    assert on.sum() == 5 * 11 * 6               # 5 days x 11 h x 6 steps per hour
    assert not on[5 * 144:].any()               # weekend off


def test_schedule_mask_with_weekend():
    on = schedule_mask(week_index(), weekday=(7.0, 24.0), weekend=(7.0, 24.0))
    assert on.sum() == 7 * 17 * 6


def test_simulation_matches_exact_solution():
    on = np.ones(13, bool)                       # 2 hours with the HVAC on
    T = simulate_temperature(on, 20.0, WARM, COOL)
    t = np.arange(13) / 6
    assert np.allclose(T, first_order_response(t, WARM[0], WARM[1], 20.0))


def test_simulation_settles_to_unheated_level_when_off():
    T = simulate_temperature(np.zeros(144 * 3, bool), 24.0, WARM, COOL)
    assert T[-1] == pytest.approx(COOL[0], abs=0.01)
    assert np.all(np.diff(T) <= 0)               # cools monotonically, no oscillation


def test_arrival_onset_finds_the_rise():
    index = week_index()[:144]                   # Monday
    hours = np.arange(144) / 6
    co2 = pd.Series(np.where(hours >= 9.0, 520.0, 450.0), index=index)
    onset = arrival_onset(co2)
    assert onset.iloc[0] == pytest.approx(9.0, abs=1 / 6)


def test_occupied_mask_and_metrics():
    index = week_index()[:144]
    day = pd.Timestamp("2024-05-06")
    occ = occupied_mask(index, pd.Series({day: 9.0}), pd.Series({day: 17.0}))
    assert occ.sum() == 8 * 6
    T = np.full(144, 23.0)                       # 2 degC above "unheated" all day
    m = schedule_metrics(np.ones(144, bool), T, occ, comfort=22.5, unheated=21.0)
    assert m["occupied_below_comfort_pct"] == 0
    assert m["degree_h_per_week"] == pytest.approx(2.0 * (143 / 6) * 7)   # trapezoid of a constant