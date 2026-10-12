from __future__ import annotations

import pandas as pd

from air_alert.modeling import load_artifact, make_design_matrix


def predict_latest(dataset: pd.DataFrame, artifact_path: str) -> pd.DataFrame:
    """Predict the latest complete timestamp for every available region."""
    required = {"timestamp", "region_id"}
    missing = required.difference(dataset.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(sorted(missing))}")

    frame = dataset.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    latest = frame["timestamp"].max()
    rows = frame.loc[frame["timestamp"].eq(latest)].copy()
    if rows.empty:
        raise ValueError("Dataset has no rows at its latest timestamp")

    artifact = load_artifact(artifact_path)
    matrix = make_design_matrix(
        rows,
        artifact["feature_columns"],
        artifact["region_columns"],
    )
    probability = artifact["model"].predict_proba(matrix)[:, 1]
    threshold = float(artifact["threshold"])
    result = pd.DataFrame(
        {
            "prediction_timestamp": latest,
            "region_id": rows["region_id"].to_numpy(),
            "probability": probability,
            "prediction": (probability >= threshold).astype("int8"),
            "threshold": threshold,
            "horizon_hours": artifact.get("horizon_hours", 3),
        }
    )
    result["risk_level"] = pd.cut(
        result["probability"],
        bins=[-float("inf"), 0.25, 0.50, 0.75, float("inf")],
        labels=["low", "medium", "high", "very_high"],
        right=False,
    ).astype(str)
    return result.sort_values("probability", ascending=False).reset_index(drop=True)
