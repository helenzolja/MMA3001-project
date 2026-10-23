"""Tests for mma3001_project.patterns (run with: python -m pytest)."""
import numpy as np
import pandas as pd
import pytest

from mma3001_project.patterns import (
    daily_event,
    daily_profile,
    rate_of_change,
    time_features,
)


def make_series(values, start="2024-05-06 00:00", name="temp_c"):
    """Values every 10 minutes, starting at local midnight in Melbourne."""
    index = pd.date_range(start, periods=len(values), freq="10min",
                          tz="Australia/Melbourne").tz_convert("UTC")
    return pd.Series(values, index=index, dtype=float, name=name)


def step_day(on_hour, off_hour, days=3, low=20.0, high=24.0):
    """Synthetic temperature: a 1-hour ramp up at on_hour, down at off_hour."""
    hours = np.arange(days * 144) / 6.0 % 24
    values = np.full(hours.shape, low)
    values = np.where(hours >= on_hour, np.minimum(low + (hours - on_hour) * (high - low), high), values)
    values = np.where(hours >= off_hour, np.maximum(high - (hours - off_hour) * (high - low), low), values)
    return make_series(values)


def test_rate_of_change_of_a_straight_line_is_its_slope():
    s = make_series(20 + 0.5 * np.arange(50) / 6.0)      # 0.5 degC per hour
    assert np.allclose(rate_of_change(s), 0.5)


def test_rate_of_change_rejects_missing_values():
    s = make_series([20.0, np.nan, 21.0, 21.5, 22.0, 22.5, 23.0, 23.5])
    with pytest.raises(ValueError, match="NaN"):
        rate_of_change(s)


def test_daily_event_finds_switch_on_time():
    s = step_day(on_hour=7.5, off_hour=22.0)
    on = daily_event(rate_of_change(s), 4, 12, kind="max", min_rate=0.5)
    assert len(on) == 3
    assert np.allclose(on["hour"], 8.0, atol=0.5)        # middle of the 1-h ramp


def test_daily_event_across_midnight():
    s = step_day(on_hour=7.0, off_hour=23.5)
    off = daily_event(rate_of_change(s), 20, 28, kind="min", min_rate=0.5)
    # The ramp down runs 23:30-00:30; its middle is midnight, i.e. hour 24.
    assert np.allclose(off["hour"].dropna(), 24.0, atol=0.5)


def test_quiet_day_has_no_event():
    s = make_series(np.full(144, 21.0))                    # flat all day
    on = daily_event(rate_of_change(s), 4, 12, kind="max", min_rate=0.5)
    assert on["time"].isna().all()


def test_time_features_and_profile():
    s = make_series(np.arange(144 * 7, dtype=float))      # one week from Monday
    f = time_features(s.index)
    assert f["hour"].iloc[0] == 0 and f["weekend"].sum() == 2 * 144
    profile = daily_profile(s)
    assert profile.shape == (144, 2)
    assert list(profile.columns) == ["weekday", "weekend"]