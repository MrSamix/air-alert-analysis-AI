from __future__ import annotations

import pandas as pd

from air_alert.data_preparation import (
    add_forward_target,
    build_hourly_panel,
    load_oblast_events,
)


def test_volunteer_loader_normalizes_utc_and_missing_finished_at(tmp_path):
    source = tmp_path / "alerts.csv"
    pd.DataFrame(
        {
            "region": ["м. Київ"],
            "started_at": ["2026-10-12 10:15:00+03:00"],
            "finished_at": [None],
        }
    ).to_csv(source, index=False)

    events = load_oblast_events(source)

    assert events.loc[0, "region_id"] == "м. Київ"
    assert events.loc[0, "started_at"] == pd.Timestamp("2026-10-12 07:15:00+00:00")
    assert events.loc[0, "finished_at"] == pd.Timestamp("2026-10-12 07:45:00+00:00")


def test_hourly_panel_does_not_create_future_hours():
    events = pd.DataFrame(
        {
            "region_id": ["м. Київ"],
            "started_at": [pd.Timestamp("2026-10-12 07:15:00", tz="UTC")],
            "finished_at": [pd.Timestamp("2026-10-12 09:22:00", tz="UTC")],
        }
    )

    panel = build_hourly_panel(
        events,
        start="2026-10-12",
        end="2026-10-12",
    )

    assert panel["timestamp"].min() == pd.Timestamp("2026-10-12 00:00:00+00:00")
    assert panel["timestamp"].max() == pd.Timestamp("2026-10-12 08:00:00+00:00")
    assert not panel["timestamp"].isin(
        pd.date_range("2026-10-12 09:00:00+00:00", periods=15, freq="h")
    ).any()


def test_forward_target_uses_only_next_hours():
    panel = pd.DataFrame(
        {
            "region_id": ["A"] * 5,
            "timestamp": pd.date_range("2026-10-12", periods=5, freq="h", tz="UTC"),
            "alert_started": [0, 1, 0, 0, 1],
        }
    )

    result = add_forward_target(panel, horizon_hours=3)

    assert result["target_3h"].tolist()[:2] == [1, 1]
    assert result["target_3h"].isna().sum() == 3
