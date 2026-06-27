from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MulticlassTemperatureScaler:
    """Single-scalar temperature scaling fit by NLL on a validation set."""

    temperature_: float = 1.0

    def fit(
        self,
        logits: np.ndarray,
        gold_indices: np.ndarray,
        search_low: float = 0.05,
        search_high: float = 5.0,
        num_candidates: int = 200,
    ) -> "MulticlassTemperatureScaler":
        logits = np.asarray(logits, dtype=np.float64)
        gold_indices = np.asarray(gold_indices, dtype=int)
        candidates = np.linspace(search_low, search_high, num_candidates)

        best_temperature = 1.0
        best_nll = float("inf")
        for temperature in candidates:
            nll = _mean_negative_log_likelihood(logits / temperature, gold_indices)
            if nll < best_nll:
                best_nll = nll
                best_temperature = float(temperature)

        self.temperature_ = best_temperature
        return self

    def predict_probabilities(self, logits: np.ndarray) -> np.ndarray:
        scaled = np.asarray(logits, dtype=np.float64) / self.temperature_
        return _softmax(scaled)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=-1, keepdims=True)


def _mean_negative_log_likelihood(logits: np.ndarray, gold_indices: np.ndarray) -> float:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    log_normalizer = np.log(np.exp(shifted).sum(axis=-1))
    gold_logits = shifted[np.arange(len(gold_indices)), gold_indices]
    return float(np.mean(log_normalizer - gold_logits))
