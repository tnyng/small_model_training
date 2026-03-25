import numpy as np

from modernbert_uq.calibration import (
    FeatureLogisticCalibrator,
    TemperatureScaler,
)


def test_calibrators_fit_and_predict_probabilities():
    logits = np.array([-1.5, -0.8, 0.2, 1.4, 2.1], dtype=float)
    labels = np.array([0, 0, 1, 1, 1], dtype=float)
    difficulty = np.array([0.8, 0.6, 0.5, 0.3, 0.2], dtype=float)

    baseline = FeatureLogisticCalibrator(use_difficulty=False).fit(
        logits=logits,
        labels=labels,
    )
    temperature = TemperatureScaler().fit(logits=logits, labels=labels)
    difficulty_aware = FeatureLogisticCalibrator(use_difficulty=True).fit(
        logits=logits,
        labels=labels,
        predicted_difficulty=difficulty,
    )

    baseline_probs = baseline.predict_proba(logits=logits)
    temperature_probs = temperature.predict_proba(logits=logits)
    difficulty_probs = difficulty_aware.predict_proba(
        logits=logits, predicted_difficulty=difficulty
    )

    assert np.all((baseline_probs >= 0.0) & (baseline_probs <= 1.0))
    assert np.all((temperature_probs >= 0.0) & (temperature_probs <= 1.0))
    assert np.all((difficulty_probs >= 0.0) & (difficulty_probs <= 1.0))
