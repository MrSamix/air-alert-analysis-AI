from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score


def build_threshold_table(
    predictions: pd.DataFrame,
    thresholds: list[float],
) -> pd.DataFrame:
    rows = []
    actual = predictions["actual"].astype("int8")
    for threshold in thresholds:
        predicted = (predictions["probability"] >= threshold).astype("int8")
        false_positive = int(((actual == 0) & (predicted == 1)).sum())
        false_negative = int(((actual == 1) & (predicted == 0)).sum())
        days = predictions["timestamp"].dt.date.nunique()
        rows.append(
            {
                "threshold": threshold,
                "precision": precision_score(actual, predicted, zero_division=0),
                "recall": recall_score(actual, predicted, zero_division=0),
                "f1": f1_score(actual, predicted, zero_division=0),
                "predicted_positive_rate": predicted.mean(),
                "false_positives": false_positive,
                "false_negatives": false_negative,
                "false_positives_per_day": false_positive / days,
                "false_negatives_per_day": false_negative / days,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare alert prediction operating points by probability threshold."
    )
    parser.add_argument("predictions_path")
    parser.add_argument("--output-path", default="reports/analysis/threshold_analysis.csv")
    args = parser.parse_args()

    predictions = pd.read_csv(args.predictions_path, parse_dates=["timestamp"])
    predictions["timestamp"] = pd.to_datetime(predictions["timestamp"], utc=True)
    thresholds = [0.20, 0.25, 0.30, 0.35, 0.3871, 0.40, 0.45, 0.50, 0.60, 0.70]
    table = build_threshold_table(predictions, thresholds)
    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
