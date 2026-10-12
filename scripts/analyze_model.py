from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)


def _period_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    y = frame["actual"].astype(int)
    probability = frame["probability"]
    prediction = frame["prediction"].astype(int)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {
        "rows": int(len(frame)),
        "positive_rate": float(y.mean()),
        "predicted_positive_rate": float(prediction.mean()),
        "pr_auc": (
            float(average_precision_score(y, probability))
            if y.nunique() == 2
            else float("nan")
        ),
        "roc_auc": (
            float(roc_auc_score(y, probability))
            if y.nunique() == 2
            else float("nan")
        ),
        "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def _save_error_analysis(predictions: pd.DataFrame, output_dir: Path) -> dict:
    predictions = predictions.copy()
    predictions["date"] = predictions["timestamp"].dt.date.astype(str)
    predictions["error_type"] = "true_negative"
    predictions.loc[
        (predictions.actual == 1) & (predictions.prediction == 1), "error_type"
    ] = "true_positive"
    predictions.loc[
        (predictions.actual == 0) & (predictions.prediction == 1), "error_type"
    ] = "false_positive"
    predictions.loc[
        (predictions.actual == 1) & (predictions.prediction == 0), "error_type"
    ] = "false_negative"

    daily = (
        predictions.groupby("date", as_index=False)
        .agg(
            actual=("actual", "sum"),
            predicted=("prediction", "sum"),
            false_positives=("error_type", lambda values: (values == "false_positive").sum()),
            false_negatives=("error_type", lambda values: (values == "false_negative").sum()),
        )
    )
    daily.to_csv(output_dir / "daily_error_analysis.csv", index=False)

    regional_rows = []
    for region, group in predictions.groupby("region_id", sort=True):
        row = _period_metrics(group)
        row["region_id"] = region
        regional_rows.append(row)
    by_region = pd.DataFrame(regional_rows)
    by_region.to_csv(output_dir / "regional_error_analysis.csv", index=False)

    cm = confusion_matrix(predictions["actual"], predictions["prediction"], labels=[0, 1])
    fig, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(cm, cmap="Blues")
    fig.colorbar(image, ax=axis)
    axis.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Predicted 0", "Predicted 1"],
        yticklabels=["Actual 0", "Actual 1"],
        title="Test confusion matrix",
    )
    for row in range(2):
        for column in range(2):
            axis.text(column, row, str(cm[row, column]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    return {
        "test": _period_metrics(predictions),
        "daily": daily.to_dict(orient="records"),
        "regions": by_region.to_dict(orient="records"),
    }


def _save_prediction_plots(predictions: pd.DataFrame, output_dir: Path) -> None:
    daily = predictions.groupby("timestamp", as_index=False).agg(
        actual=("actual", "mean"), probability=("probability", "mean")
    )
    fig, axis = plt.subplots(figsize=(12, 4))
    axis.plot(daily["timestamp"], daily["actual"], label="Actual target rate")
    axis.plot(daily["timestamp"], daily["probability"], label="Mean predicted probability")
    axis.set(title="Actual versus predicted risk by hour", ylabel="Rate / probability")
    axis.legend()
    axis.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_dir / "actual_vs_predicted.png", dpi=150)
    plt.close(fig)

    precision, recall, _ = precision_recall_curve(
        predictions["actual"], predictions["probability"]
    )
    fig, axis = plt.subplots(figsize=(6, 5))
    axis.plot(recall, precision)
    axis.set(xlabel="Recall", ylabel="Precision", title="Test precision-recall curve")
    axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "precision_recall_curve.png", dpi=150)
    plt.close(fig)


def _season_summary(dataset: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    frame = dataset.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.loc[frame["target_3h"].notna()].copy()
    frame["period"] = "summer"
    frame.loc[frame["timestamp"] >= "2026-09-01", "period"] = "sep_oct"
    summary = {}
    for period, group in frame.groupby("period"):
        summary[period] = {
            "rows": int(len(group)),
            "regions": int(group["region_id"].nunique()),
            "target_positive_rate": float(group["target_3h"].mean()),
            "alert_started_rate": float(group["alert_started"].mean()),
            "alert_active_rate": float(group["alert_active"].mean()),
            "alerts_started_total": int(group["alert_started"].sum()),
        }
    return summary


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Create plots and drift/error reports.")
    parser.add_argument("dataset_path")
    parser.add_argument("predictions_path")
    parser.add_argument("--output-dir", default="reports/analysis")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_csv(args.predictions_path, parse_dates=["timestamp"])
    predictions["timestamp"] = pd.to_datetime(predictions["timestamp"], utc=True)
    dataset = pd.read_csv(args.dataset_path)

    report = {
        "season_summary": _season_summary(dataset),
        "error_analysis": _save_error_analysis(predictions, output_dir),
    }
    _save_prediction_plots(predictions, output_dir)
    (output_dir / "drift_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
