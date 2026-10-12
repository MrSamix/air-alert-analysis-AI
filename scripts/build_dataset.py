from __future__ import annotations

import argparse

from air_alert.data_preparation import prepare_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert raw air-alert CSV into an hourly oblast-level dataset."
    )
    parser.add_argument("input_path", help="Path to the raw CSV file")
    parser.add_argument("output_path", help="Path for the processed CSV file")
    parser.add_argument(
        "--horizon-hours",
        type=int,
        default=3,
        help="Forecast horizon in complete future hours (default: 3)",
    )
    parser.add_argument(
        "--start-date",
        help="Inclusive UTC start date, for example 2026-05-01",
    )
    parser.add_argument(
        "--end-date",
        help="Inclusive UTC end date, for example 2026-10-11",
    )
    args = parser.parse_args()
    if (args.start_date is None) != (args.end_date is None):
        parser.error("--start-date and --end-date must be provided together")
    prepare_dataset(
        args.input_path,
        args.output_path,
        horizon_hours=args.horizon_hours,
        start=args.start_date,
        end=args.end_date,
    )
    print(
        "The requested end date is an upper bound; the processed dataset "
        "stops before the hour containing the last observed event."
    )


if __name__ == "__main__":
    main()
