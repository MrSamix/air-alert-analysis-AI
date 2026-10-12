from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from air_alert.inference import predict_latest


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description=(
            "Predict alert risk from the latest fully observed UTC hour in "
            "a processed dataset."
        )
    )
    parser.add_argument("dataset_path")
    parser.add_argument("artifact_path")
    parser.add_argument("output_path")
    parser.add_argument(
        "--as-of",
        help=(
            "Forecast wall-clock time in UTC. The latest usable timestamp is "
            "the last complete UTC hour before this time; defaults to now."
        ),
    )
    args = parser.parse_args()

    predictions = predict_latest(
        pd.read_csv(args.dataset_path),
        args.artifact_path,
        as_of=args.as_of,
    )
    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output, index=False)
    print(predictions.to_string(index=False))


if __name__ == "__main__":
    main()
