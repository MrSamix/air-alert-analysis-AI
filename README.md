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

For development and tests:

```powershell
python -m pip install -e ".[dev]"
```

## Automated tests

Run the complete test suite from the repository root:

```powershell
$env:PYTHONPATH = "src"
python -m pytest -q
```

Run a specific test file:

```powershell
$env:PYTHONPATH = "src"
python -m pytest -q tests/test_data_preparation.py
```

The tests cover UTC normalization, Kyiv preservation, the observed-hour
dataset cutoff, forward-target boundaries, leakage-safe feature selection,
chronological split boundaries, and inference timestamp semantics. The
current suite contains 7 tests.

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
- The strict start-of-hour model does not use the current-hour
  `alert_active` value. It uses only lagged and rolling `alert_active`
  history, whose latest observation is strictly before the prediction hour.
- Dates selected in the UI are passed to the same preparation functions as
  the CLI.
- Uploading a file does not automatically replace the existing raw dataset;
  click **Save uploaded CSV** explicitly.
- Build the processed dataset again whenever the raw CSV changes.
- `--end-date` is only an upper bound. The builder caps the hourly grid before
  the hour containing the last observed `started_at`/`finished_at` timestamp
  in the raw CSV. This excludes a possibly incomplete current hour and avoids
  creating empty future hours merely because a later calendar date was
  selected.
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

### Streamlit screenshots

The following screenshots show the main workflow in the local Streamlit
application:

**Application overview**

![Streamlit application overview](./docs/streamlit-overview.png)

**Building the dataset and configuring the chronological split**

![Streamlit dataset and training controls](./docs/streamlit-training.png)

**Generating the latest forecast with an explicit UTC as-of time**

![Streamlit forecast controls](./docs/streamlit-forecast.png)

**Reviewing saved metrics and reports**

![Streamlit reports](./docs/streamlit-reports.png)

These images document the interface layout. The displayed local paths and
timestamps depend on the machine and dataset used when the screenshots were
captured.

## Example latest forecast

The latest generated per-region forecast is available in
[reports/latest_predictions.csv](./reports/latest_predictions.csv).

This report was generated from the fully observed hour
`2026-10-12 07:00 UTC` and covers the next three hours:

```text
forecast start: 2026-10-12 08:00 UTC
forecast end:   2026-10-12 11:00 UTC (exclusive)
threshold:      0.3541
```

Selected rows from the report:

| Region | Probability | Prediction | Risk level |
|---|---:|---:|---|
| Харківська область | 0.983 | 1 | very_high |
| Дніпропетровська область | 0.976 | 1 | very_high |
| Миколаївська область | 0.939 | 1 | very_high |
| Київська область | 0.931 | 1 | very_high |
| Сумська область | 0.925 | 1 | very_high |
| м. Київ | 0.629 | 1 | high |
| Волинська область | 0.487 | 1 | medium |
| Тернопільська область | 0.275 | 0 | medium |
| Львівська область | 0.110 | 0 | low |
| Закарпатська область | 0.005 | 0 | low |

`prediction = 1` means that the probability is at least the selected
threshold; it does not mean that an alert is guaranteed. The `probability`
is a model estimate for at least one new alert start during the forecast
window, not a prediction of exact alert time or duration. The model output is
analytical and must not replace official air-alert notifications.

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

Using the current leakage-safe pipeline, dataset, and fixed baseline split:

```text
Train:      before 2026-09-20
Validation: 2026-09-20 through 2026-09-30
Test:       2026-10-01 through the latest valid target timestamp
```

the current test metrics are:

```text
PR-AUC:    0.9124
ROC-AUC:   0.8868
Precision: 0.7961
Recall:    0.9113
LogLoss:   0.4123
Brier:     0.1317
Threshold: 0.3541
```

These numbers are dataset- and split-specific. They are not a guarantee of
future operational performance. The test target is available only through
`2026-10-12 05:00 UTC` because a three-hour forward target requires three
subsequent observed hours.

For historical comparison, the previous pipeline version reported PR-AUC
`0.9149`, ROC-AUC `0.8909`, precision `0.8052`, and recall `0.9078`. Those
values used the previous feature policy and are not the current baseline.

## Limitations and next steps

The current MVP is functional and includes automated tests, Streamlit
verification, leakage checks, timestamp semantics, rolling retraining, and
training-window experiments. The remaining limitations and planned
improvements are:

- Add selectable threshold modes to Streamlit:
  safety-first, balanced, precision-oriented, and custom threshold.
- Add regional metrics and regional error summaries to the main model report.
- Continue monitoring future observations and retrain with a documented
  rolling schedule.
- Repeat rolling evaluation on future test periods instead of relying on one
  historical test window.
- Persist the `as_of` timestamp and selected complete observation hour in every
  operational forecast audit record.
- Keep the fixed baseline test unchanged when comparing future model changes.
- Finalize repository inclusion rules for raw data,
  model artifacts, and generated reports.

The model is a research prototype. Its probabilities are not official
warnings, do not predict exact alert time or duration, and must not replace
official air-alert notification systems.

## Rolling retraining experiment

The repository includes an expanding-window experiment:

```powershell
$env:PYTHONPATH = "src"

python scripts/rolling_experiment.py `
    data/processed/latest_features.csv `
    --output-path reports/rolling_retraining_experiment.csv
```

Each run keeps validation and test chronological. The model is trained from
the beginning of the dataset up to a later `train_end`; the validation window
also moves forward. No test rows are used for threshold selection.

The report contains one row per run with split sizes, threshold, PR-AUC,
ROC-AUC, LogLoss, Brier score, precision, and recall. This is a monitoring
experiment, not a license to select a model on the final test period.

## Training-window experiment

To compare how much history the model should use, run:

```powershell
$env:PYTHONPATH = "src"

python scripts/training_window_experiment.py `
    data/processed/latest_features.csv `
    --output-path reports/training_window_experiment.csv
```

All four models use the same fixed split:

```text
Train end:      2026-09-20
Validation end: 2026-10-01
Test:           after 2026-10-01
```

Only the training rows change:

```text
full_history: all rows before 2026-09-20
90d:          last 90 days before 2026-09-20
60d:          last 60 days before 2026-09-20
30d:          last 30 days before 2026-09-20
```

The validation set remains responsible for threshold selection, and the test
set is used only for the final comparison.

The current results are:

| Window | Train rows | Threshold | PR-AUC | ROC-AUC | LogLoss | Brier | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Full history | 70,272 | 0.3541 | 0.9124 | **0.8868** | **0.4123** | **0.1317** | 0.7961 | **0.9113** |
| 90 days | 51,840 | 0.3631 | **0.9097** | 0.8853 | 0.4128 | 0.1317 | 0.7986 | **0.9142** |
| 60 days | 34,560 | 0.4298 | 0.9076 | 0.8812 | 0.4197 | 0.1345 | **0.8055** | 0.8983 |
| 30 days | 17,280 | 0.4065 | 0.8927 | 0.8678 | 0.4444 | 0.1423 | 0.7906 | 0.9006 |

The full-history window is the current default because it has the strongest
overall ranking and calibration metrics while retaining high recall. The
90-day window is a close alternative when a slightly higher recall is more
important. The 30-day window loses ranking and calibration quality on this
test period, so it should not replace full history without further evidence.
