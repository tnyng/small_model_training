import numpy as np

from modernbert_uq.evaluate import evaluate_predictions
from modernbert_uq.predict import export_prediction_table


def test_evaluate_predictions_returns_core_metrics():
    labels = np.array([0, 1, 1, 0], dtype=float)
    probabilities = np.array([0.1, 0.8, 0.7, 0.2], dtype=float)
    logits = np.array([-2.2, 1.4, 0.9, -1.1], dtype=float)
    difficulty = np.array([0.2, 0.9, 0.8, 0.1], dtype=float)
    difficulty_targets = np.array([0.1, 0.95, 0.75, 0.2], dtype=float)

    metrics = evaluate_predictions(
        labels=labels,
        probabilities=probabilities,
        logits=logits,
        difficulty_predictions=difficulty,
        difficulty_targets=difficulty_targets,
    )

    assert 0.0 <= metrics["classification"]["auroc"] <= 1.0
    assert "brier_score" in metrics["classification"]
    assert "ece" in metrics["classification"]
    assert "rmse" in metrics["difficulty"]
    assert "mae" in metrics["difficulty"]


def test_export_prediction_table_returns_rows_for_calibration():
    table = export_prediction_table(
        ids=["a", "b"],
        logits=np.array([1.0, -1.0]),
        probabilities=np.array([0.73, 0.27]),
        predicted_difficulty=np.array([0.2, 0.7]),
        labels=np.array([1, 0]),
    )

    assert list(table.columns) == [
        "id",
        "logit",
        "probability",
        "predicted_difficulty",
        "label",
    ]
    assert table.to_dict(orient="records")[0]["id"] == "a"
