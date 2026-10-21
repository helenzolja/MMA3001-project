"""Tests for mma3001_project.load: JSON flattening, de-duplication, stale-sensor detection."""
import json

import numpy as np
import pandas as pd
import pytest

from mma3001_project.load import (clean_sensor_frame, cleaning_report, gap_lengths,
                                  load_sensor_series, parse_env_csv, sensor_summary)

LIVE, STALE = "6012002000869", "6012002000227"


def _payload(t, co2, formaldehyde_error=False):
    items = [{"values": [{"time": t, "value": co2}],
              "variable": {"id": 67, "name": "Carbon dioxide", "unit": "ppm"}}]
    hcho = {"time": t, "errorCode": 8} if formaldehyde_error else {"time": t, "value": 20}
    items.append({"values": [hcho], "variable": {"id": 70, "name": "Formaldehyde", "unit": "µg/m3"}})
    return json.dumps(items)


@pytest.fixture
def raw_csv(tmp_path):
    """Small export mimicking the real file: live sensor uploaded twice per reading,
    stale sensor repeating one frozen reading, one reading with an errorCode."""
    t0 = 1709866800  # 2024-03-08 03:00 UTC
    rows = []
    for k in range(20):
        t = t0 + 600 * k
        for _ in range(2):  # each reading uploaded twice (gateway re-send)
            rows.append({"id": len(rows), "sensorid": LIVE, "devicetype": "env",
                         "jsondata": _payload(t, 450 + k, formaldehyde_error=(k == 3)),
                         "status": "Processed", "processing_errors": None,
                         "createdate": "2024-03-08 03:00:00+11", "processdate": None})
        rows.append({"id": len(rows), "sensorid": STALE, "devicetype": "env",
                     "jsondata": _payload(t0 - 10**7, 666), "status": "Processed",
                     "processing_errors": None, "createdate": "2024-03-08 03:00:00+11",
                     "processdate": None})
    path = tmp_path / "env.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_ids_stay_strings_and_duplicates_removed(raw_csv):
    long = parse_env_csv(raw_csv)
    assert set(long["sensorid"]) == {LIVE, STALE}  # 13 digits intact, not 6.01E+12
    co2 = long[(long.sensorid == LIVE) & (long.variable == "Carbon dioxide")]
    assert len(co2) == 20  # 40 uploads -> 20 real readings


def test_error_coded_readings_dropped(raw_csv):
    long = parse_env_csv(raw_csv)
    hcho = long[(long.sensorid == LIVE) & (long.variable == "Formaldehyde")]
    assert len(hcho) == 19


def test_stale_sensor_detected(raw_csv):
    summary = sensor_summary(parse_env_csv(raw_csv))
    assert summary.loc[LIVE, "is_live"]
    assert not summary.loc[STALE, "is_live"]


def test_stale_sensor_rejected(raw_csv):
    with pytest.raises(ValueError, match="stale"):
        load_sensor_series(parse_env_csv(raw_csv), STALE)


def test_unknown_sensor_rejected(raw_csv):
    with pytest.raises(ValueError, match="Unknown sensor"):
        load_sensor_series(parse_env_csv(raw_csv), "123")


def test_missing_columns_rejected(tmp_path):
    p = tmp_path / "bad.csv"
    pd.DataFrame({"a": [1]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="Missing required columns"):
        parse_env_csv(p)


def test_series_on_regular_grid_with_range_check(raw_csv):
    s = load_sensor_series(parse_env_csv(raw_csv), LIVE, valid_range=(350, 460))
    assert (s.index[1] - s.index[0]) == pd.Timedelta("10min")
    assert s.isna().sum() == 9  # values 461..469 are out of range -> NaN


def test_gap_lengths():
    s = pd.Series([1, np.nan, np.nan, 2, np.nan, 3, np.nan, np.nan, np.nan])
    assert gap_lengths(s).tolist() == [2, 1, 3]
    assert gap_lengths(pd.Series([1.0, 2.0])).empty


def test_clean_frame_and_report(raw_csv):
    df = clean_sensor_frame(parse_env_csv(raw_csv), LIVE, variables=("Carbon dioxide",))
    assert list(df.columns) == ["co2_ppm"]
    rep = cleaning_report(df)
    assert rep.loc["co2_ppm", "slots"] == 20 and rep.loc["co2_ppm", "missing"] == 0