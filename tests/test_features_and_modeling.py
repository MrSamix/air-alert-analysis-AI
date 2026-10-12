from __future__ import annotations

import pandas as pd

from air_alert.features import add_features
from air_alert.modeling import (
    TimeSplit,
    _feature_columns,
    chronological_split,
)


def _panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "region_id": ["A"] * 6,
            "timestamp": pd.date_range("2026-10-12", periods=6, freq="h", tz="UTC"),
            "alert_started": [0, 1, 0, 0, 1, 0],
            "alert_active": [0, 1, 1, 0, 1, 1],
        }
    )


def test_features_use_previous_alert_active_values_only():
    result = add_features(_panel())

    assert result.loc[0, "alert_active_lag_1h"] != result.loc[0, "alert_active"]
    assert pd.isna(result.loc[0, "alert_active_lag_1h"])
    assert result.loc[2, "alert_active_lag_1h"] == 1
    assert result.loc[2, "alert_active_lag_2h"] == 0
    assert result.loc[2, "alert_active_sum_3h"] == 1


def test_current_alert_active_is_excluded_from_model_features():
    frame = add_features(_panel())
    frame["target_3h"] = 0

    columns = _feature_columns(frame)

    assert "alert_active" not in columns
    assert "alert_active_lag_1h" in columns
    assert "target_3h" not in columns
    assert "alert_started" not in columns


def test_chronological_split_has_no_overlap():
    frame = add_features(_panel())
    frame["target_3h"] = [0, 1, 1, 0, 1, 0]

    split = chronological_split(frame, "2026-10-12T02:00:00Z", "2026-10-12T04:00:00Z")

    assert split.train["timestamp"].max() < split.validation["timestamp"].min()
    assert split.validation["timestamp"].max() < split.test["timestamp"].min()
