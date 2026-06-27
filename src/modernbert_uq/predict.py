from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def export_prediction_table(
    ids: Sequence[str],
    predicted_difficulty: np.ndarray,
    raw_score_outputs: np.ndarray | None = None,
    score_predictions: np.ndarray | None = None,
    labels: np.ndarray | None = None,
    score_task_type: str = "classification",
    logits: np.ndarray | None = None,
    probabilities: np.ndarray | None = None,
) -> pd.DataFrame:
    if raw_score_outputs is None:
        raw_score_outputs = logits
    if score_predictions is None:
        score_predictions = probabilities
    if raw_score_outputs is None or score_predictions is None:
        raise ValueError("Both raw_score_outputs/logits and score_predictions/probabilities are required")
    frame = pd.DataFrame(
        {
            "id": list(ids),
            "raw_score_output": np.asarray(raw_score_outputs, dtype=float),
            "score_prediction": np.asarray(score_predictions, dtype=float),
            "predicted_difficulty": np.asarray(predicted_difficulty, dtype=float),
        }
    )
    if labels is not None:
        if score_task_type == "classification":
            frame = frame.rename(
                columns={
                    "raw_score_output": "logit",
                    "score_prediction": "probability",
                }
            )
            frame["label"] = np.asarray(labels)
        else:
            frame["score_target"] = np.asarray(labels, dtype=float)
    return frame
