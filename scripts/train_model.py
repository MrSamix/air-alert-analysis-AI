from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from air_alert.modeling import chronological_split, train_classifier


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train an air-alert classifier using chronological validation."
    )
    parser.add_argument("dataset_path")
    parser.add_argument("train_end", help="Exclusive train end, e.g. 2026-09-01")
    parser.add_argument("validation_end", help="Exclusive validation end")
    parser.add_argument("--report-path", default="reports/metrics.json")
    parser.add_argument("--predictions-path", default="reports/test_predictions.csv")
    args = parser.parse_args()

    dataset = pd.read_csv(args.dataset_path)
    split = chronological_split(dataset, args.train_end, args.validation_end)
    result = train_classifier(split)

    report = {
        "train_rows": len(split.train),
        "validation_rows": len(split.validation),
        "test_rows": len(split.test),
        "threshold": result["threshold"],
        "validation_metrics": result["validation_metrics"],
        "test_metrics": result["test_metrics"],
    }
    report_path = Path(args.report_path)
    predictions_path = Path(args.predictions_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    result["test_predictions"].to_csv(predictions_path, index=False)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
