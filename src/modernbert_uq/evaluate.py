from __future__ import annotations

import math

import numpy as np


def evaluate_predictions(
    labels: np.ndarray,
    probabilities: np.ndarray,
    logits: np.ndarray,
    difficulty_predictions: np.ndarray | None = None,
    difficulty_targets: np.ndarray | None = None,
) -> dict[str, dict[str, float]]:
    labels = np.asarray(labels, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    logits = np.asarray(logits, dtype=float)

    results = {
        "classification": compute_classification_metrics(labels, probabilities, logits)
    }
    if difficulty_predictions is not None and difficulty_targets is not None:
        results["difficulty"] = compute_regression_metrics(
            np.asarray(difficulty_targets, dtype=float),
            np.asarray(difficulty_predictions, dtype=float),
        )
    return results


def compute_classification_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    logits: np.ndarray,
) -> dict[str, float]:
    labels = labels.astype(int)
    predicted_labels = (probabilities >= 0.5).astype(int)
    return {
        "auroc": float(binary_auroc(labels, probabilities)),
        "auprc": float(binary_average_precision(labels, probabilities)),
        "accuracy": float(np.mean(labels == predicted_labels)),
        "f1": float(binary_f1(labels, predicted_labels)),
        "log_loss": float(binary_log_loss(labels, probabilities)),
        "brier_score": float(np.mean(np.square(probabilities - labels))),
        "ece": float(expected_calibration_error(labels, probabilities)),
        "mean_logit": float(np.mean(logits)),
    }


def compute_regression_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, float]:
    errors = predictions - targets
    rmse = math.sqrt(float(np.mean(np.square(errors))))
    mae = float(np.mean(np.abs(errors)))
    correlation = float(np.corrcoef(targets, predictions)[0, 1])
    return {"rmse": rmse, "mae": mae, "pearson": correlation}


def expected_calibration_error(
    labels: np.ndarray, probabilities: np.ndarray, num_bins: int = 10
) -> float:
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    total = len(labels)
    ece = 0.0
    for lower, upper in zip(bins[:-1], bins[1:]):
        if upper == 1.0:
            mask = (probabilities >= lower) & (probabilities <= upper)
        else:
            mask = (probabilities >= lower) & (probabilities < upper)
        if not np.any(mask):
            continue
        bin_accuracy = np.mean(labels[mask])
        bin_confidence = np.mean(probabilities[mask])
        ece += (np.sum(mask) / total) * abs(bin_accuracy - bin_confidence)
    return float(ece)


def binary_log_loss(labels: np.ndarray, probabilities: np.ndarray) -> float:
    clipped = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
    return float(-np.mean(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped)))


def binary_f1(labels: np.ndarray, predictions: np.ndarray) -> float:
    true_positive = np.sum((labels == 1) & (predictions == 1))
    false_positive = np.sum((labels == 0) & (predictions == 1))
    false_negative = np.sum((labels == 1) & (predictions == 0))
    denom = (2 * true_positive) + false_positive + false_negative
    if denom == 0:
        return float("nan")
    return float((2 * true_positive) / denom)


def binary_auroc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    positives = probabilities[labels == 1]
    negatives = probabilities[labels == 0]
    if len(positives) == 0 or len(negatives) == 0:
        return float("nan")

    wins = 0.0
    for pos in positives:
        wins += np.sum(pos > negatives)
        wins += 0.5 * np.sum(pos == negatives)
    return float(wins / (len(positives) * len(negatives)))


def binary_average_precision(labels: np.ndarray, probabilities: np.ndarray) -> float:
    positive_count = int(np.sum(labels == 1))
    if positive_count == 0:
        return float("nan")

    order = np.argsort(-probabilities)
    sorted_labels = labels[order]
    cumulative_true = np.cumsum(sorted_labels == 1)
    ranks = np.arange(1, len(sorted_labels) + 1)
    precision_at_k = cumulative_true / ranks
    return float(np.sum(precision_at_k[sorted_labels == 1]) / positive_count)
