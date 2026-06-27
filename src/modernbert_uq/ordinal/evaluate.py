from __future__ import annotations

import numpy as np

from modernbert_uq.distribution.evaluate import evaluate_distribution_predictions


def evaluate_ordinal_predictions(
    gold_scores: np.ndarray,
    class_probabilities: np.ndarray,
    cumulative_gt: np.ndarray,
    expected_scores: np.ndarray,
    argmax_scores: np.ndarray,
    threshold_scores: np.ndarray,
    label_offset: int,
    difficulty_predictions: np.ndarray | None = None,
    difficulty_targets: np.ndarray | None = None,
) -> dict[str, dict[str, float]]:
    base = evaluate_distribution_predictions(
        gold_scores=gold_scores,
        class_probabilities=class_probabilities,
        expected_scores=expected_scores,
        argmax_scores=argmax_scores,
        label_offset=label_offset,
        difficulty_predictions=difficulty_predictions,
        difficulty_targets=difficulty_targets,
    )

    cumulative_gt = np.asarray(cumulative_gt, dtype=float)
    threshold_scores = np.asarray(threshold_scores, dtype=float)
    gold_scores = np.asarray(gold_scores, dtype=float)

    monotonic_violations = 0
    if cumulative_gt.shape[1] >= 2:
        diffs = np.diff(cumulative_gt, axis=-1)
        monotonic_violations = int(np.sum(diffs > 1e-6))

    threshold_exact_match = float(np.mean(np.rint(threshold_scores) == np.rint(gold_scores)))
    threshold_mae = float(np.mean(np.abs(threshold_scores - gold_scores)))

    base["score_ordinal"] = {
        "monotonic_violation_count": float(monotonic_violations),
        "threshold_accuracy": threshold_exact_match,
        "threshold_mae": threshold_mae,
    }
    return base
