from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from air_alert.inference import predict_latest


class FixedProbabilityModel:
    def predict_proba(self, matrix):
        return np.tile([0.2, 0.8], (len(matrix), 1))


def test_predict_latest_uses_last_complete_hour_and_forecast_window(tmp_path):
    timestamps = pd.date_range(
        "2026-10-12 07:00:00+00:00", periods=4, freq="h"
    )
    dataset = pd.DataFrame(
        {
            "timestamp": timestamps,
            "region_id": ["м. Київ"] * 4,
            "hour_sin": [0.0] * 4,
        }
    )
    artifact_path = tmp_path / "model.pkl"
    with artifact_path.open("wb") as file:
        pickle.dump(
            {
                "model": FixedProbabilityModel(),
                "feature_columns": ["hour_sin"],
                "region_columns": ["м. Київ"],
                "threshold": 0.5,
                "horizon_hours": 3,
            },
            file,
        )

    result = predict_latest(
        dataset,
        str(artifact_path),
        as_of="2026-10-12T10:30:00+00:00",
    )

    assert result["prediction_timestamp"].iloc[0] == pd.Timestamp(
        "2026-10-12 09:00:00+00:00"
    )
    assert result["forecast_start_timestamp"].iloc[0] == pd.Timestamp(
        "2026-10-12 10:00:00+00:00"
    )
    assert result["forecast_end_timestamp"].iloc[0] == pd.Timestamp(
        "2026-10-12 13:00:00+00:00"
    )
    assert result["prediction"].iloc[0] == 1
