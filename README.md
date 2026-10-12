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
latest available hourly timestamp. `horizon_hours=3` means the forecast covers
the following three complete hours.

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
