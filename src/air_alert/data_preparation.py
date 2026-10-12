from __future__ import annotations

from pathlib import Path

import pandas as pd
import numpy as np


REQUIRED_COLUMNS = {
    "oblast",
    "raion",
    "hromada",
    "level",
    "started_at",
    "finished_at",
    "source",
}
VOLUNTEER_REQUIRED_COLUMNS = {"region", "started_at", "finished_at"}


def _parse_boundary(value: str | pd.Timestamp | None, name: str) -> pd.Timestamp | None:
    if value is None:
        return None
    boundary = pd.Timestamp(value)
    if boundary.tzinfo is None:
        boundary = boundary.tz_localize("UTC")
    else:
        boundary = boundary.tz_convert("UTC")
    if name == "end" and len(str(value)) == 10:
        boundary += pd.Timedelta(days=1)
    return boundary


def load_oblast_events(
    path: str | Path,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Load oblast alert intervals from official or volunteer CSV format."""
    events = pd.read_csv(path)
    if VOLUNTEER_REQUIRED_COLUMNS.issubset(events.columns):
        events = events.rename(columns={"region": "region_id"}).copy()
        events["source"] = "volunteer"
        if "naive" in events.columns:
            events["naive"] = events["naive"].fillna(False).astype(bool)
        else:
            events["naive"] = False
    else:
        missing = REQUIRED_COLUMNS.difference(events.columns)
        if missing:
            raise ValueError(
                f"CSV is missing required columns for known formats: {', '.join(sorted(missing))}"
            )
        events = events.loc[
            events["level"].astype("string").str.strip().str.lower().isin(
                {"oblast", "raion", "hromada"}
            )
        ].copy()
        events["region_id"] = events["oblast"].astype("string").str.strip()
        events["source"] = events["source"].fillna("official")
        events["naive"] = False

    events["region_id"] = (
        events["region_id"].astype("string").str.strip().replace(
            {"Київ": "м. Київ", "м.Київ": "м. Київ"}
        )
    )
    events["started_at"] = pd.to_datetime(events["started_at"], utc=True, errors="coerce")
    events["finished_at"] = pd.to_datetime(
        events["finished_at"], utc=True, errors="coerce"
    )
    missing_finished = events["finished_at"].isna()
    events.loc[missing_finished, "finished_at"] = (
        events.loc[missing_finished, "started_at"] + pd.Timedelta(minutes=30)
    )
    events = events.dropna(subset=["region_id", "started_at"])
    events = events.loc[
        events["finished_at"].isna()
        | events["finished_at"].ge(events["started_at"])
    ]
    start_at = _parse_boundary(start, "start")
    end_at = _parse_boundary(end, "end")
    if start_at is not None and end_at is not None and start_at >= end_at:
        raise ValueError("start must be earlier than end")
    if start_at is not None:
        events = events.loc[
            events["finished_at"].isna() | events["finished_at"].gt(start_at)
        ]
    if end_at is not None:
        events = events.loc[events["started_at"].lt(end_at)]

    # Identical source rows are not separate alerts. Overlapping hierarchy
    # rows are handled as one active regional signal in build_hourly_panel.
    result = events.drop_duplicates(
        subset=["region_id", "started_at", "finished_at", "source"]
    ).reset_index(drop=True)
    result.attrs["period_start"] = start_at
    result.attrs["period_end"] = end_at
    return result


def build_hourly_panel(
    events: pd.DataFrame,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Convert alert intervals into one row per oblast and UTC hour.

    The requested end date is an upper bound only. The hourly panel never
    extends beyond the last observed event timestamp, preventing synthetic
    future hours when a date range is longer than the raw CSV.
    """
    required = {"region_id", "started_at", "finished_at"}
    missing = required.difference(events.columns)
    if missing:
        raise ValueError(
            f"Events are missing required columns: {', '.join(sorted(missing))}"
        )
    if events.empty:
        raise ValueError("No valid oblast-level events remain after preprocessing")

    start_at = _parse_boundary(start, "start")
    if start_at is None:
        start_at = events["started_at"].min().floor("h")
    requested_end = _parse_boundary(end, "end")
    observed_end = events[["started_at", "finished_at"]].max().max()
    if pd.isna(observed_end):
        raise ValueError("Events have no valid observed timestamp")
    # The hour containing the last event may still be in progress in the raw
    # feed. Use its start as an exclusive boundary, so the panel contains only
    # fully observed hours before it.
    observed_end = observed_end.floor("h")
    end_at = observed_end if requested_end is None else min(requested_end, observed_end)
    if start_at >= end_at:
        raise ValueError("start must be earlier than end")
    hours = pd.date_range(
        start_at.floor("h"), end_at.ceil("h") - pd.Timedelta(hours=1), freq="h", tz="UTC"
    )
    regions = pd.Index(events["region_id"].dropna().unique(), name="region_id")
    index = pd.MultiIndex.from_product(
        [regions, hours], names=["region_id", "timestamp"]
    )
    panel = pd.DataFrame(index=index).reset_index()

    starts = events.assign(
        timestamp=events["started_at"].dt.floor("h"), alert_started=1
    )[["region_id", "timestamp", "alert_started"]].drop_duplicates(
        ["region_id", "timestamp"]
    )
    panel = panel.merge(starts, on=["region_id", "timestamp"], how="left")
    panel["alert_started"] = panel["alert_started"].fillna(0).astype("int8")

    panel["alert_active"] = 0
    hour_start = hours[0]
    hour_count = len(hours)
    for region, region_events in events.groupby("region_id", sort=False):
        changes = np.zeros(hour_count + 1, dtype="int32")
        for event in region_events.itertuples(index=False):
            first_hour = max(
                0, int((event.started_at.floor("h") - hour_start).total_seconds() // 3600)
            )
            if pd.isna(event.finished_at):
                last_hour = hour_count
            else:
                last_hour = int(
                    (event.finished_at.ceil("h") - hour_start).total_seconds() // 3600
                )
            first_hour = min(first_hour, hour_count)
            last_hour = min(max(last_hour, first_hour), hour_count)
            if first_hour < last_hour:
                changes[first_hour] += 1
                changes[last_hour] -= 1
        active = np.cumsum(changes[:-1]) > 0
        panel.loc[panel["region_id"].eq(region), "alert_active"] = active.astype("int8")

    return panel.sort_values(["region_id", "timestamp"]).reset_index(drop=True)


def add_forward_target(panel: pd.DataFrame, horizon_hours: int = 3) -> pd.DataFrame:
    """Add whether a new alert starts in the next H complete hours."""
    if horizon_hours < 1:
        raise ValueError("horizon_hours must be at least 1")
    required = {"region_id", "timestamp", "alert_started"}
    missing = required.difference(panel.columns)
    if missing:
        raise ValueError(
            f"Panel is missing required columns: {', '.join(sorted(missing))}"
        )

    result = panel.sort_values(["region_id", "timestamp"]).copy()
    future_starts = [
        result.groupby("region_id", sort=False)["alert_started"].shift(-offset)
        for offset in range(1, horizon_hours + 1)
    ]
    result[f"target_{horizon_hours}h"] = (
        pd.concat(future_starts, axis=1).max(axis=1, skipna=False).astype("Int64")
    )
    return result


def prepare_dataset(
    input_path: str | Path,
    output_path: str | Path,
    horizon_hours: int = 3,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    add_model_features: bool = True,
) -> None:
    """Run CSV-to-model-table preparation."""
    events = load_oblast_events(input_path, start=start, end=end)
    if events.empty:
        raise ValueError("No oblast-level events overlap the requested period")
    panel = build_hourly_panel(events, start=start, end=end)
    dataset = add_forward_target(panel, horizon_hours=horizon_hours)
    if add_model_features:
        from air_alert.features import add_features

        dataset = add_features(dataset)
    dataset.to_csv(output_path, index=False)
