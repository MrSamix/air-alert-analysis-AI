from __future__ import annotations

import pandas as pd

from air_alert.modeling import load_artifact, make_design_matrix


def _last_complete_hour(as_of: str | pd.Timestamp | None = None) -> pd.Timestamp:
    """Return the last UTC hour whose observations are complete."""
    if as_of is None:
        current = pd.Timestamp.now(tz="UTC")
    else:
        current = pd.Timestamp(as_of)
        if current.tzinfo is None:
            current = current.tz_localize("UTC")
        else:
            current = current.tz_convert("UTC")
    return current.floor("h") - pd.Timedelta(hours=1)


def predict_latest(
    dataset: pd.DataFrame,
    artifact_path: str,
    as_of: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Predict the next horizon from the latest fully observed UTC hour.

    ``as_of`` is the wall-clock time at which the forecast is made. If omitted,
    the current UTC time is used. The current, still-in-progress hour is never
    used as the prediction timestamp.
    """
    required = {"timestamp", "region_id"}
    missing = required.difference(dataset.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(sorted(missing))}")

    frame = dataset.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    cutoff = _last_complete_hour(as_of)
    available = frame.loc[frame["timestamp"].le(cutoff), "timestamp"]
    if available.empty:
        raise ValueError(
            "Dataset has no fully observed hourly timestamp at or before "
            f"{cutoff.isoformat()}"
        )
    latest = available.max()
    rows = frame.loc[frame["timestamp"].eq(latest)].copy()
    if rows.empty:
        raise ValueError("Dataset has no rows at the selected complete timestamp")

    artifact = load_artifact(artifact_path)
    horizon_hours = int(artifact.get("horizon_hours", 3))
    matrix = make_design_matrix(
        rows,
        artifact["feature_columns"],
        artifact["region_columns"],
    )
    probability = artifact["model"].predict_proba(matrix)[:, 1]
    threshold = float(artifact["threshold"])
    forecast_start = latest + pd.Timedelta(hours=1)
    forecast_end = latest + pd.Timedelta(hours=horizon_hours + 1)
    result = pd.DataFrame(
        {
            "prediction_timestamp": latest,
            "forecast_start_timestamp": forecast_start,
            "forecast_end_timestamp": forecast_end,
            "region_id": rows["region_id"].to_numpy(),
            "probability": probability,
            "prediction": (probability >= threshold).astype("int8"),
            "threshold": threshold,
            "horizon_hours": horizon_hours,
        }
    )
    result["risk_level"] = pd.cut(
        result["probability"],
        bins=[-float("inf"), 0.25, 0.50, 0.75, float("inf")],
        labels=["low", "medium", "high", "very_high"],
        right=False,
    ).astype(str)
    return result.sort_values("probability", ascending=False).reset_index(drop=True)
