from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from air_alert.modeling import chronological_split, train_classifier


DEFAULT_RUNS = (
    ("run_1", "2026-09-20", "2026-10-01"),
    ("run_2", "2026-09-27", "2026-10-04"),
    ("run_3", "2026-10-01", "2026-10-08"),
)


def run_experiment(dataset: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for run_id, train_end, validation_end in DEFAULT_RUNS:
        split = chronological_split(dataset, train_end, validation_end)
        result = train_classifier(split)
        metrics = result["test_metrics"]
        rows.append(
            {
                "run_id": run_id,
                "train_end": train_end,
                "validation_end": validation_end,
                "test_start": validation_end,
                "test_end": split.test["timestamp"].max(),
                "train_rows": len(split.train),
                "validation_rows": len(split.validation),
                "test_rows": len(split.test),
                "threshold": result["threshold"],
                **{f"test_{key}": value for key, value in metrics.items()},
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare expanding-window chronological retraining runs."
    )
    parser.add_argument("dataset_path")
    parser.add_argument(
        "--output-path", default="reports/rolling_retraining_experiment.csv"
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
