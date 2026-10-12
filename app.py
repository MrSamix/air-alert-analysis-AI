from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from air_alert.data_preparation import prepare_dataset
from air_alert.inference import predict_latest
from air_alert.modeling import chronological_split, save_artifact, train_classifier


ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"
MODELS_DIR = ROOT / "models"


def _path_input(label: str, default: Path) -> Path:
    value = st.text_input(label, value=str(default))
    return Path(value)


def main() -> None:
    st.set_page_config(page_title="Air Alert Analysis", page_icon="🛡️", layout="wide")
    st.title("Air Alert Analysis")
    st.caption(
        "Research prototype: forecast of a new alert start in the next three hours."
    )
    st.warning(
        "This is not an official warning system. Use official air-alert sources "
        "for real-world safety decisions."
    )

    with st.sidebar:
        st.header("Dataset")
        upload = st.file_uploader("Upload raw CSV", type=["csv"])
        input_path = _path_input("Or use existing CSV", RAW_DIR / "alerts.csv")
        start_date = st.date_input("Start date", value=pd.Timestamp("2026-05-21"))
        end_date = st.date_input("End date", value=pd.Timestamp("2026-10-11"))
        horizon = st.number_input("Forecast horizon (hours)", min_value=1, max_value=24, value=3)
        if upload is not None:
            input_path = RAW_DIR / upload.name
            if st.button("Save uploaded CSV"):
                RAW_DIR.mkdir(parents=True, exist_ok=True)
                input_path.write_bytes(upload.getvalue())
                st.success(f"Saved: {input_path}")

    st.subheader("1. Build dataset")
    processed_path = _path_input(
        "Processed dataset path", PROCESSED_DIR / "latest_features.csv"
    )
    if st.button("Build hourly features", type="primary"):
        try:
            prepare_dataset(
                input_path,
                processed_path,
                horizon_hours=int(horizon),
                start=start_date.isoformat(),
                end=end_date.isoformat(),
            )
            st.success(f"Dataset created: {processed_path}")
            st.dataframe(pd.read_csv(processed_path).tail(24), use_container_width=True)
        except (OSError, ValueError, pd.errors.ParserError) as error:
            st.error(f"Dataset build failed: {error}")

    st.subheader("2. Train model")
    train_end = st.date_input("Train end (exclusive)", value=pd.Timestamp("2026-09-20"))
    validation_end = st.date_input(
        "Validation end (exclusive)", value=pd.Timestamp("2026-10-01")
    )
    artifact_path = _path_input("Model artifact path", MODELS_DIR / "air_alert_model.pkl")
    metrics_path = _path_input("Metrics report path", REPORTS_DIR / "metrics.json")
    test_predictions_path = _path_input(
        "Test predictions path", REPORTS_DIR / "test_predictions.csv"
    )

    if st.button("Train and save model", type="primary"):
        try:
            dataset = pd.read_csv(processed_path)
            split = chronological_split(dataset, train_end.isoformat(), validation_end.isoformat())
            result = train_classifier(split)
            save_artifact(result, artifact_path, horizon_hours=int(horizon))
            report = {
                "train_rows": len(split.train),
                "validation_rows": len(split.validation),
                "test_rows": len(split.test),
                "threshold": result["threshold"],
                "validation_metrics": result["validation_metrics"],
                "test_metrics": result["test_metrics"],
                "artifact_path": str(artifact_path),
                "horizon_hours": int(horizon),
            }
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            test_predictions_path.parent.mkdir(parents=True, exist_ok=True)
            metrics_path.write_text(
                json.dumps(report, indent=2), encoding="utf-8"
            )
            result["test_predictions"].to_csv(test_predictions_path, index=False)
            st.success(f"Model saved: {artifact_path}")
            st.json(report)
        except (OSError, ValueError, KeyError) as error:
            st.error(f"Training failed: {error}")

    st.subheader("3. Forecast latest timestamp")
    as_of = st.datetime_input(
        "Forecast as-of time (UTC)",
        value=pd.Timestamp.now(tz="UTC").to_pydatetime(),
        help=(
            "The current hour is excluded because it is not complete. "
            "Leave this at the current UTC time for an operational forecast."
        ),
    )
    latest_predictions_path = _path_input(
        "Latest predictions path", REPORTS_DIR / "latest_predictions.csv"
    )
    if st.button("Generate latest forecast", type="primary"):
        try:
            predictions = predict_latest(
                pd.read_csv(processed_path),
                str(artifact_path),
                as_of=as_of,
            )
            latest_predictions_path.parent.mkdir(parents=True, exist_ok=True)
            predictions.to_csv(latest_predictions_path, index=False)
            st.success(f"Forecast saved: {latest_predictions_path}")
            st.metric("Forecast timestamp", str(predictions["prediction_timestamp"].iloc[0]))
            st.dataframe(predictions, use_container_width=True)
        except (OSError, ValueError, KeyError) as error:
            st.error(f"Forecast failed: {error}")

    st.subheader("4. Existing reports")
    if metrics_path.exists():
        st.json(json.loads(metrics_path.read_text(encoding="utf-8")))
    if latest_predictions_path.exists():
        st.dataframe(pd.read_csv(latest_predictions_path), use_container_width=True)


if __name__ == "__main__":
    main()
