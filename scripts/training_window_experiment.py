from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from air_alert.modeling import TimeSplit, chronological_split, train_classifier


WINDOWS = (("full_history", None), ("90d", 90), ("60d", 60), ("30d", 30))


def run_experiment(
    dataset: pd.DataFrame,
    train_end: str = "2026-09-20",
    validation_end: str = "2026-10-01",
) -> pd.DataFrame:
    baseline = chronological_split(dataset, train_end, validation_end)
    train_cutoff = pd.Timestamp(train_end, tz="UTC")
    rows: list[dict[str, object]] = []

    for window_name, window_days in WINDOWS:
        if window_days is None:
            train = baseline.train
        else:
            window_start = train_cutoff - pd.Timedelta(days=window_days)
            train = baseline.train.loc[
                baseline.train["timestamp"] >= window_start
            ].copy()
        split = TimeSplit(train=train, validation=baseline.validation, test=baseline.test)
        result = train_classifier(split)
        rows.append(
            {
                "training_window": window_name,
                "train_start": train["timestamp"].min(),
                "train_end": train_end,
                "validation_start": baseline.validation["timestamp"].min(),
                "validation_end": validation_end,
                "test_start": baseline.test["timestamp"].min(),
                "test_end": baseline.test["timestamp"].max(),
                "train_rows": len(train),
                "validation_rows": len(baseline.validation),
                "test_rows": len(baseline.test),
                "threshold": result["threshold"],
                **{f"test_{key}": value for key, value in result["test_metrics"].items()},
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare 30-day, 60-day, 90-day, and full-history training "
            "windows on one fixed chronological test split."
        )
    )
    parser.add_argument("dataset_path")
    parser.add_argument(
        "--output-path", default="reports/training_window_experiment.csv"
    )
    args = parser.parse_args()

    dataset = pd.read_csv(args.dataset_path)
    dataset["timestamp"] = pd.to_datetime(dataset["timestamp"], utc=True)
    report = run_experiment(dataset)
    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output, index=False)
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
