# Air Alert Analysis

An MVP research pipeline for forecasting whether a new air-raid alert will
start in a Ukrainian oblast during the next three hours.

> This is an analytical research prototype. It is not an official warning
> system and must not replace official air-alert notifications.

## Problem definition

The model produces one prediction per region and hourly timestamp:

```text
features at T -> probability of at least one new alert start at T+1, T+2, or T+3
```

The current model covers 24 regions, including `м. Київ`, and uses UTC
timestamps internally. The current production-like artifact uses the volunteer
oblast-level dataset without the `naive` column.

## Leakage policy

- Train, validation, and test periods are strictly chronological.
- No random shuffle is used.
- Lag, rolling, and EWMA features use values strictly before the prediction
  timestamp.
- The target uses future alert starts only and is never used as a feature.
- Region columns and the feature schema are learned from training data and
  stored with the model artifact.

## Repository structure

```text
data/raw/          source CSV files
data/processed/    hourly feature datasets
src/air_alert/     preparation, features, modeling, inference
scripts/           dataset, training, prediction, and analysis commands
models/            serialized model artifacts
reports/           metrics, predictions, plots, and drift reports
```

## Installation

```powershell
python -m pip install -e .
```

## Streamlit interface

The project includes a Streamlit UI in [app.py](./app.py). It is intended for
interactive exploration and demo usage, while the CLI scripts remain the
reproducible and automation-friendly interface.

### Start the application

Install the project dependencies:

```powershell
python -m pip install -e .
```

Start Streamlit from the repository root:

```powershell
streamlit run app.py
```

The application opens in a browser, usually at:

```text
http://localhost:8501
```

### UI workflow

#### 1. Select the input data

In the sidebar, either:

- upload a raw CSV and click **Save uploaded CSV**; or
- enter the path to an existing CSV, for example:

  ```text
  data/raw/alerts.csv
  ```

Supported raw formats are documented in the
[dataset preparation section](#build-a-processed-dataset).

#### 2. Build hourly features

Set:

- start date;
- inclusive end date;
- forecast horizon in hours.

Then click **Build hourly features**. The UI creates the processed hourly
dataset with:

- one row per region and hour;
- `alert_started`;
- `alert_active`;
- the forward target for training;
- lag, rolling, EWMA, and calendar features.

The processed path can be changed from the default:

```text
data/processed/latest_features.csv
```

#### 3. Train and save the model

Set the exclusive boundaries:

```text
Train end:      2026-09-20
Validation end: 2026-10-01
```

The split is chronological:

```text
train:      timestamp < train_end
validation: train_end <= timestamp < validation_end
test:       timestamp >= validation_end
```

Click **Train and save model**. The UI writes:

- model artifact, by default to `models/air_alert_model.pkl`;
- metrics, by default to `reports/metrics.json`;
- test predictions, by default to `reports/test_predictions.csv`.

The saved artifact includes the model, feature schema, training regions,
threshold, and forecast horizon.

#### 4. Generate the latest forecast

Click **Generate latest forecast** after training. The UI loads the saved
artifact and predicts from the latest fully observed hour at the selected
UTC as-of time.

The output contains:

```text
prediction_timestamp
forecast_start_timestamp
forecast_end_timestamp
region_id
probability
prediction
threshold
horizon_hours
risk_level
```

The result is saved by default to:

```text
reports/latest_predictions.csv
```

For example, with `horizon_hours=3` and a selected prediction timestamp of
`23:00 UTC`, the forecast concerns the following three complete hours:

```text
00:00–03:00 UTC
03:00–06:00 Europe/Kyiv
```

### Streamlit notes

- The UI uses UTC internally, as does the CLI pipeline.
- Dates selected in the UI are passed to the same preparation functions as
  the CLI.
- Uploading a file does not automatically replace the existing raw dataset;
  click **Save uploaded CSV** explicitly.
- Build the processed dataset again whenever the raw CSV changes.
- Retrain the model when the feature dataset or training period changes.
- The latest forecast is based on the last fully observed timestamp at the
  selected UTC as-of time. Rows later than that cutoff are ignored.
- The model output is an analytical probability and must not replace official
  air-alert notifications.

### Troubleshooting

If `streamlit` is not recognized, install the project dependencies again:

```powershell
python -m pip install -e .
```

If the application reports that a model artifact is missing, train the model
first or enter the correct artifact path. If predictions use an old timestamp,
rebuild the processed dataset with the updated raw CSV.

## Build a processed dataset

The loader supports both:

```text
region,started_at,finished_at
region,started_at,finished_at,naive
```

and the older hierarchical official format.

```powershell
$env:PYTHONPATH = "src"

python scripts/build_dataset.py `
    data/raw/alerts.csv `
    data/processed/latest_features.csv `
    --start-date 2026-05-21 `
    --end-date 2026-10-11 `
    --horizon-hours 3
```

## Train and save the model

```powershell
$env:PYTHONPATH = "src"

python scripts/train_model.py `
    data/processed/latest_features.csv `
    2026-09-20 `
    2026-10-01 `
    --report-path reports/metrics.json `
    --predictions-path reports/test_predictions.csv `
    --artifact-path models/air_alert_model.pkl `
    --horizon-hours 3
```

## Generate the latest forecast

```powershell
$env:PYTHONPATH = "src"

python scripts/predict.py `
    data/processed/latest_features.csv `
    models/air_alert_model.pkl `
    reports/latest_predictions.csv
```

The output contains a probability and binary decision for each region at the
latest fully observed hourly timestamp. By default, the CLI uses the current
UTC time as `as_of` and excludes the current, still-in-progress hour.
`horizon_hours=3` means the forecast covers the following three complete hours.

To reproduce a forecast for a fixed historical cutoff:

```powershell
python scripts/predict.py `
    data/processed/latest_features.csv `
    models/air_alert_model.pkl `
    reports/latest_predictions.csv `
    --as-of 2026-10-12T15:32:00+00:00
```

For this example, the latest usable prediction timestamp is `2026-10-12
14:00 UTC`; the forecast covers `15:00–18:00 UTC`. The output records:

```text
prediction_timestamp       last complete observation hour
forecast_start_timestamp   first future hour being forecast
forecast_end_timestamp     exclusive end of the forecast window
```

This distinction prevents a processed dataset with rows extended to a future
`--end-date` from silently producing a forecast from an incomplete or
not-yet-observed hour. The same rule is applied by the Streamlit interface via
its **Forecast as-of time (UTC)** control.

## Analysis

```powershell
$env:PYTHONPATH = "src"

python scripts/analyze_model.py `
    data/processed/latest_features.csv `
    reports/test_predictions.csv `
    --output-dir reports/analysis

python scripts/threshold_analysis.py `
    reports/test_predictions.csv `
    --output-path reports/analysis/threshold_analysis.csv
```

Current test results are stored in `reports/metrics.json`. They should be
updated whenever the dataset, split, or model artifact changes.

## Current model result

Using the current dataset and the split:

```text
Train:      before 2026-09-20
Validation: 2026-09-20 through 2026-09-30
Test:       2026-10-01 through 2026-10-11
```

the current test metrics are:

```text
PR-AUC:    0.9149
ROC-AUC:   0.8909
Precision: 0.8052
Recall:    0.9078
LogLoss:   0.4056
Brier:     0.1292
```

These numbers are dataset- and split-specific. They are not a guarantee of
future operational performance.

## Limitations and next steps

- Validate the model on future data not used for threshold selection.
- Monitor concept drift and retrain using a documented rolling window.
- Compare full-history, 30-day, 60-day, and 90-day training windows.
- Calibrate thresholds for safety-first and precision-oriented use cases.
- Report metrics by region, not only globally.
- Add automated tests for leakage and inference schema compatibility.
- Preserve the `as_of` timestamp and selected complete hour with every
  operational forecast.
