from __future__ import annotations

import math

import numpy as np

from modernbert_uq.evaluate import (
    compute_regression_metrics,
    expected_calibration_error,
    rankdata,
)


def evaluate_distribution_predictions(
    gold_scores: np.ndarray,
    class_probabilities: np.ndarray,
    expected_scores: np.ndarray,
    argmax_scores: np.ndarray,
    label_offset: int,
    difficulty_predictions: np.ndarray | None = None,
    difficulty_targets: np.ndarray | None = None,
) -> dict[str, dict[str, float]]:
    gold_scores = np.asarray(gold_scores, dtype=float)
    class_probabilities = np.asarray(class_probabilities, dtype=float)
    expected_scores = np.asarray(expected_scores, dtype=float)
    argmax_scores = np.asarray(argmax_scores, dtype=float)

    num_classes = class_probabilities.shape[-1]
    gold_indices = np.rint(gold_scores - label_offset).astype(int)
    if np.any(gold_indices < 0) or np.any(gold_indices >= num_classes):
        raise ValueError(
            "Gold scores are outside the configured label range. "
            f"Expected values in [{label_offset}, {label_offset + num_classes - 1}]."
        )

    results: dict[str, dict[str, float]] = {
        "score_regression": compute_regression_metrics(gold_scores, expected_scores),
        "score_classification": _classification_metrics(
            gold_indices=gold_indices,
            argmax_scores=argmax_scores,
            label_offset=label_offset,
            num_classes=num_classes,
        ),
        "score_distribution": _distribution_metrics(
            gold_indices=gold_indices,
            class_probabilities=class_probabilities,
        ),
    }
    if difficulty_predictions is not None and difficulty_targets is not None:
        results["difficulty"] = compute_regression_metrics(
            np.asarray(difficulty_targets, dtype=float),
            np.asarray(difficulty_predictions, dtype=float),
        )
    return results


def _classification_metrics(
    gold_indices: np.ndarray,
    argmax_scores: np.ndarray,
    label_offset: int,
    num_classes: int,
) -> dict[str, float]:
    predicted_indices = np.rint(argmax_scores - label_offset).astype(int)
    accuracy = float(np.mean(gold_indices == predicted_indices))
    macro_f1 = float(_macro_f1(gold_indices, predicted_indices, num_classes))
    kappa = float(_quadratic_weighted_kappa(gold_indices, predicted_indices, num_classes))
    return {"accuracy": accuracy, "macro_f1": macro_f1, "quadratic_weighted_kappa": kappa}


def _distribution_metrics(
    gold_indices: np.ndarray,
    class_probabilities: np.ndarray,
) -> dict[str, float]:
    clipped = np.clip(class_probabilities, 1e-12, 1.0)
    picked = clipped[np.arange(len(gold_indices)), gold_indices]
    nll = float(-np.mean(np.log(picked)))

    row_entropy = -np.sum(clipped * np.log(clipped), axis=-1)
    predictive_entropy = float(np.mean(row_entropy))

    num_classes = class_probabilities.shape[-1]
    one_hot = np.zeros_like(class_probabilities)
    one_hot[np.arange(len(gold_indices)), gold_indices] = 1.0
    brier = float(np.mean(np.sum((class_probabilities - one_hot) ** 2, axis=-1)))

    top1_probability = class_probabilities.max(axis=-1)
    top1_correct = (class_probabilities.argmax(axis=-1) == gold_indices).astype(float)
    ece = float(expected_calibration_error(top1_correct, top1_probability))

    rank_gold = rankdata(gold_indices.astype(float))
    expected_from_probs = (
        class_probabilities
        * np.arange(num_classes, dtype=float)
    ).sum(axis=-1)
    rank_expected = rankdata(expected_from_probs)
    if len(rank_gold) <= 1:
        spearman_expected = float("nan")
    else:
        spearman_expected = float(np.corrcoef(rank_gold, rank_expected)[0, 1])

    return {
        "nll": nll,
        "predictive_entropy": predictive_entropy,
        "brier_score": brier,
        "ece_top1": ece,
        "spearman_expected_vs_gold_rank": spearman_expected,
    }


def _macro_f1(
    gold_indices: np.ndarray,
    predicted_indices: np.ndarray,
    num_classes: int,
) -> float:
    f1_sum = 0.0
    counted = 0
    for class_index in range(num_classes):
        gold_is_class = gold_indices == class_index
        pred_is_class = predicted_indices == class_index
        true_positive = int(np.sum(gold_is_class & pred_is_class))
        false_positive = int(np.sum((~gold_is_class) & pred_is_class))
        false_negative = int(np.sum(gold_is_class & (~pred_is_class)))
        denom = (2 * true_positive) + false_positive + false_negative
        if denom == 0:
            continue
        f1_sum += (2.0 * true_positive) / denom
        counted += 1
    if counted == 0:
        return float("nan")
    return f1_sum / counted


def _quadratic_weighted_kappa(
    gold_indices: np.ndarray,
    predicted_indices: np.ndarray,
    num_classes: int,
) -> float:
    if len(gold_indices) == 0:
        return float("nan")
    confusion = np.zeros((num_classes, num_classes), dtype=float)
    for g, p in zip(gold_indices, predicted_indices):
        confusion[int(g), int(p)] += 1.0

    row_totals = confusion.sum(axis=1)
    col_totals = confusion.sum(axis=0)
    expected = np.outer(row_totals, col_totals) / max(confusion.sum(), 1.0)

    weights = np.zeros_like(confusion)
    denom = max((num_classes - 1) ** 2, 1)
    for i in range(num_classes):
        for j in range(num_classes):
            weights[i, j] = ((i - j) ** 2) / denom

    numerator = float(np.sum(weights * confusion))
    denominator = float(np.sum(weights * expected))
    if denominator == 0.0:
        return float("nan")
    return 1.0 - (numerator / denominator)


__all__ = ["evaluate_distribution_predictions"]
