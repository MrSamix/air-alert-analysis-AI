from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class TimeSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def chronological_split(
    dataset: pd.DataFrame,
    train_end: str | pd.Timestamp,
    validation_end: str | pd.Timestamp,
) -> TimeSplit:
    """Split by timestamp boundaries; rows are never shuffled."""
    if "timestamp" not in dataset or "target_3h" not in dataset:
        raise ValueError("Dataset must contain timestamp and target_3h")
    frame = dataset.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    train_at = pd.Timestamp(train_end)
    validation_at = pd.Timestamp(validation_end)
    if train_at.tzinfo is None:
        train_at = train_at.tz_localize("UTC")
    if validation_at.tzinfo is None:
        validation_at = validation_at.tz_localize("UTC")
    if train_at >= validation_at:
        raise ValueError("train_end must be earlier than validation_end")

    frame = frame.loc[frame["target_3h"].notna()].sort_values("timestamp")
    result = TimeSplit(
        train=frame.loc[frame["timestamp"] < train_at].copy(),
        validation=frame.loc[
            (frame["timestamp"] >= train_at) & (frame["timestamp"] < validation_at)
        ].copy(),
        test=frame.loc[frame["timestamp"] >= validation_at].copy(),
    )
    if any(part.empty for part in (result.train, result.validation, result.test)):
        raise ValueError("Chronological split produced an empty partition")
    return result


def _feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {"timestamp", "target_3h", "alert_started"}
    return [
        column
        for column in frame.columns
        if column not in excluded and column != "region_id"
    ]


def make_design_matrices(
    split: TimeSplit,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Create numeric matrices and fit category columns from train only."""
    train_regions = sorted(split.train["region_id"].dropna().unique())
    feature_columns = _feature_columns(split.train)

    def transform(frame: pd.DataFrame) -> pd.DataFrame:
        return make_design_matrix(frame, feature_columns, train_regions)

    return (
        transform(split.train),
        transform(split.validation),
        transform(split.test),
        feature_columns,
    )


def make_design_matrix(
    frame: pd.DataFrame,
    feature_columns: list[str],
    train_regions: list[str],
) -> pd.DataFrame:
    """Transform rows using the exact feature and region schema from training."""
    missing = set(feature_columns) - set(frame.columns)
    if missing:
        raise ValueError(
            f"Dataset is missing model features: {', '.join(sorted(missing))}"
        )
    numeric = frame[feature_columns].apply(pd.to_numeric, errors="coerce")
    regions = pd.get_dummies(frame["region_id"], prefix="region", dtype="int8")
    regions = regions.reindex(
        columns=[f"region_{region}" for region in train_regions], fill_value=0
    )
    return pd.concat(
        [numeric.reset_index(drop=True), regions.reset_index(drop=True)], axis=1
    )


def save_artifact(result: dict[str, Any], path: str | Path, horizon_hours: int = 3) -> None:
    """Persist the trained model and the exact inference schema."""
    artifact = {
        "model": result["model"],
        "feature_columns": result["feature_columns"],
        "region_columns": result["region_columns"],
        "threshold": result["threshold"],
        "horizon_hours": horizon_hours,
        "model_type": "HistGradientBoostingClassifier",
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as file:
        pickle.dump(artifact, file, protocol=pickle.HIGHEST_PROTOCOL)


def load_artifact(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as file:
        artifact = pickle.load(file)
    required = {"model", "feature_columns", "region_columns", "threshold"}
    missing = required.difference(artifact)
    if missing:
        raise ValueError(f"Model artifact is missing: {', '.join(sorted(missing))}")
    return artifact


def _safe_metric(metric: Any, y_true: pd.Series, probabilities: np.ndarray) -> float:
    if y_true.nunique() < 2:
        return float("nan")
    return float(metric(y_true, probabilities))


def evaluate_predictions(
    y_true: pd.Series,
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    predictions = (probabilities >= threshold).astype("int8")
    return {
        "pr_auc": _safe_metric(average_precision_score, y_true, probabilities),
        "roc_auc": _safe_metric(roc_auc_score, y_true, probabilities),
        "log_loss": float(log_loss(y_true, probabilities, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, probabilities)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
    }


def choose_threshold(y_true: pd.Series, probabilities: np.ndarray) -> float:
    """Choose the F1 threshold on validation only."""
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if thresholds.size == 0:
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    return float(thresholds[int(np.nanargmax(f1))])


def train_classifier(split: TimeSplit) -> dict[str, Any]:
    x_train, x_validation, x_test, feature_columns = make_design_matrices(split)
    train_regions = sorted(split.train["region_id"].dropna().unique())
    y_train = split.train["target_3h"].astype("int8")
    y_validation = split.validation["target_3h"].astype("int8")
    y_test = split.test["target_3h"].astype("int8")

    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=42,
    )
    model.fit(x_train, y_train)
    validation_probability = model.predict_proba(x_validation)[:, 1]
    threshold = choose_threshold(y_validation, validation_probability)
    test_probability = model.predict_proba(x_test)[:, 1]
    return {
        "model": model,
        "feature_columns": feature_columns,
        "region_columns": train_regions,
        "threshold": threshold,
        "validation_metrics": evaluate_predictions(
            y_validation, validation_probability, threshold
        ),
        "test_metrics": evaluate_predictions(y_test, test_probability, threshold),
        "test_predictions": pd.DataFrame(
            {
                "timestamp": split.test["timestamp"].to_numpy(),
                "region_id": split.test["region_id"].to_numpy(),
                "actual": y_test.to_numpy(),
                "probability": test_probability,
                "prediction": (test_probability >= threshold).astype("int8"),
            }
        ),
    }
