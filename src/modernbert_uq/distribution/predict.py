from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def export_distribution_prediction_table(
    ids: Sequence[str],
    class_probabilities: np.ndarray,
    expected_scores: np.ndarray,
    argmax_scores: np.ndarray,
    predictive_entropy: np.ndarray,
    gold_scores: np.ndarray | None = None,
    predicted_difficulty: np.ndarray | None = None,
    gold_difficulty: np.ndarray | None = None,
    label_offset: int = 1,
    accuracy_scores: np.ndarray | None = None,
) -> pd.DataFrame:
    class_probabilities = np.asarray(class_probabilities, dtype=float)
    num_classes = class_probabilities.shape[-1]

    columns: dict[str, np.ndarray | list[str]] = {
        "id": list(ids),
        "expected_score": np.asarray(expected_scores, dtype=float),
        "argmax_score": np.asarray(argmax_scores, dtype=float),
        "predictive_entropy": np.asarray(predictive_entropy, dtype=float),
    }
    for class_index in range(num_classes):
        column_name = f"p_{class_index + label_offset}"
        columns[column_name] = class_probabilities[:, class_index]

    frame = pd.DataFrame(columns)

    if gold_scores is not None:
        frame["score_bucket"] = np.asarray(gold_scores, dtype=float)
    if predicted_difficulty is not None:
        frame["predicted_difficulty"] = np.asarray(predicted_difficulty, dtype=float)
    if gold_difficulty is not None:
        frame["gold_difficulty"] = np.asarray(gold_difficulty, dtype=float)
    if accuracy_scores is not None:
        frame["accuracy_score"] = np.asarray(accuracy_scores, dtype=float)
    return frame
