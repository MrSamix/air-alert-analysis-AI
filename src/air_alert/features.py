from __future__ import annotations

import numpy as np
import pandas as pd


BASE_COLUMNS = {"region_id", "timestamp", "alert_started", "alert_active"}
LAGS = (1, 2, 3, 6, 12, 24)
ROLLING_WINDOWS = (3, 6, 12, 24, 72, 168, 336)


def _group_shift(panel: pd.DataFrame, column: str, periods: int = 1) -> pd.Series:
    return panel.groupby("region_id", sort=False)[column].shift(periods)


def add_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Add features that use only observations strictly before each timestamp."""
    missing = BASE_COLUMNS.difference(panel.columns)
    if missing:
        raise ValueError(f"Panel is missing required columns: {', '.join(sorted(missing))}")

    result = panel.sort_values(["region_id", "timestamp"]).reset_index(drop=True).copy()
    grouped = result.groupby("region_id", sort=False)

    timestamp = pd.to_datetime(result["timestamp"], utc=True)
    hour = timestamp.dt.hour
    day_of_week = timestamp.dt.dayofweek
    result["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    result["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    result["dow_sin"] = np.sin(2 * np.pi * day_of_week / 7)
    result["dow_cos"] = np.cos(2 * np.pi * day_of_week / 7)
    result["is_weekend"] = (day_of_week >= 5).astype("int8")

    for column in ("alert_started", "alert_active"):
        history = grouped[column].shift(1)
        for lag in LAGS:
            result[f"{column}_lag_{lag}h"] = grouped[column].shift(lag)
        for window in ROLLING_WINDOWS:
            result[f"{column}_sum_{window}h"] = (
                history.groupby(result["region_id"], sort=False)
                .rolling(window, min_periods=1)
                .sum()
                .reset_index(level=0, drop=True)
            )
            result[f"{column}_mean_{window}h"] = (
                history.groupby(result["region_id"], sort=False)
                .rolling(window, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )
        result[f"{column}_ewm_24h"] = (
            history.groupby(result["region_id"], sort=False)
            .ewm(span=24, adjust=False, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

    last_start = result["timestamp"].where(result["alert_started"].gt(0))
    last_start = last_start.groupby(result["region_id"], sort=False).ffill()
    last_start = last_start.groupby(result["region_id"], sort=False).shift(1)
    elapsed = (result["timestamp"] - last_start).dt.total_seconds() / 3600
    result["has_previous_alert_start"] = last_start.notna().astype("int8")
    result["hours_since_last_alert_start"] = elapsed.fillna(-1.0)

    return result
