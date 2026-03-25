from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TemperatureScaler:
    temperature_: float = 1.0

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> "TemperatureScaler":
        logits = np.asarray(logits, dtype=float)
        labels = np.asarray(labels, dtype=float)

        candidates = np.linspace(0.05, 5.0, 200)
        best_temperature = 1.0
        best_loss = float("inf")
        for candidate in candidates:
            probabilities = _sigmoid(logits / candidate)
            candidate_loss = binary_log_loss(labels, probabilities)
            if candidate_loss < best_loss:
                best_loss = candidate_loss
                best_temperature = float(candidate)

        self.temperature_ = best_temperature
        return self

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        logits = np.asarray(logits, dtype=float)
        return _sigmoid(logits / self.temperature_)


@dataclass
class FeatureLogisticCalibrator:
    use_difficulty: bool = False
    weights_: np.ndarray | None = None
    bias_: float = 0.0

    def fit(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        predicted_difficulty: np.ndarray | None = None,
    ) -> "FeatureLogisticCalibrator":
        features = build_features(logits, predicted_difficulty, self.use_difficulty)
        labels = np.asarray(labels, dtype=float)

        weights = np.zeros(features.shape[1], dtype=float)
        bias = 0.0
        learning_rate = 0.1
        l2_penalty = 1e-4
        sample_count = features.shape[0]

        for _ in range(2000):
            linear = features @ weights + bias
            probs = _sigmoid(linear)
            error = probs - labels
            grad_weights = (features.T @ error) / sample_count + (l2_penalty * weights)
            grad_bias = float(np.mean(error))
            weights -= learning_rate * grad_weights
            bias -= learning_rate * grad_bias

        self.weights_ = weights
        self.bias_ = bias
        return self

    def predict_proba(
        self, logits: np.ndarray, predicted_difficulty: np.ndarray | None = None
    ) -> np.ndarray:
        if self.weights_ is None:
            raise ValueError("Calibrator must be fitted before prediction")
        features = build_features(logits, predicted_difficulty, self.use_difficulty)
        return _sigmoid(features @ self.weights_ + self.bias_)


def build_features(
    logits: np.ndarray,
    predicted_difficulty: np.ndarray | None,
    use_difficulty: bool,
) -> np.ndarray:
    logits = np.asarray(logits, dtype=float).reshape(-1, 1)
    if not use_difficulty:
        return logits
    if predicted_difficulty is None:
        raise ValueError("predicted_difficulty is required for difficulty-aware calibration")
    difficulty = np.asarray(predicted_difficulty, dtype=float).reshape(-1, 1)
    return np.concatenate([logits, difficulty], axis=1)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return 1.0 / (1.0 + np.exp(-values))


def binary_log_loss(labels: np.ndarray, probabilities: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=float)
    clipped = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1.0 - 1e-6)
    return float(-np.mean(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped)))
