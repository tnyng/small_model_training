from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def export_prediction_table(
    ids: Sequence[str],
    logits: np.ndarray,
    probabilities: np.ndarray,
    predicted_difficulty: np.ndarray,
    labels: np.ndarray | None = None,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "id": list(ids),
            "logit": np.asarray(logits, dtype=float),
            "probability": np.asarray(probabilities, dtype=float),
            "predicted_difficulty": np.asarray(predicted_difficulty, dtype=float),
        }
    )
    if labels is not None:
        frame["label"] = np.asarray(labels)
    return frame
