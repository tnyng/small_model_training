import pytest

pytestmark = pytest.mark.requires_torch


def test_end_to_end_pipeline_creates_artifacts(tmp_path):
    from types import SimpleNamespace

    import torch
    import torch.nn as nn
    import accelerate  # noqa: F401

    from modernbert_uq.calibration import FeatureLogisticCalibrator
    from modernbert_uq.modeling import ModernBertUQModel
    from modernbert_uq.predict import export_prediction_table
    from modernbert_uq.train import TrainConfig, predict_batches, run_training

    class DummyEncoder(nn.Module):
        def __init__(self, hidden_size: int):
            super().__init__()
            self.config = SimpleNamespace(hidden_size=hidden_size)
            self.embedding = nn.Embedding(128, hidden_size)

        def forward(self, input_ids, attention_mask=None):
            hidden = self.embedding(input_ids)
            if attention_mask is not None:
                hidden = hidden * attention_mask.unsqueeze(-1)
            return SimpleNamespace(last_hidden_state=hidden)

    def build_batch(input_ids, labels, difficulty):
        attention_mask = (input_ids != 0).long()
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "difficulty": difficulty,
            "label_mask": torch.ones_like(labels, dtype=torch.bool),
            "difficulty_mask": torch.ones_like(difficulty, dtype=torch.bool),
        }

    model = ModernBertUQModel(encoder=DummyEncoder(hidden_size=10), dropout_prob=0.2)
    train_batches = [
        build_batch(
            torch.tensor([[1, 2, 3], [4, 5, 0]], dtype=torch.long),
            torch.tensor([1.0, 0.0]),
            torch.tensor([0.2, 0.7]),
        )
    ]
    eval_batches = [
        build_batch(
            torch.tensor([[2, 2, 2], [3, 3, 0]], dtype=torch.long),
            torch.tensor([1.0, 0.0]),
            torch.tensor([0.3, 0.6]),
        )
    ]

    result = run_training(
        model,
        train_batches,
        eval_batches,
        TrainConfig(
            learning_rate=1e-3,
            num_epochs=1,
            gradient_accumulation_steps=1,
            output_dir=tmp_path,
        ),
    )

    predictions = predict_batches(model, eval_batches)
    table = export_prediction_table(
        ids=["eval-0", "eval-1"],
        logits=predictions["logits"],
        probabilities=predictions["probabilities"],
        predicted_difficulty=predictions["difficulty_predictions"],
        labels=predictions["labels"],
    )
    calibrator = FeatureLogisticCalibrator(use_difficulty=True).fit(
        logits=table["logit"].to_numpy(),
        labels=table["label"].to_numpy(),
        predicted_difficulty=table["predicted_difficulty"].to_numpy(),
    )
    calibrated = calibrator.predict_proba(
        logits=table["logit"].to_numpy(),
        predicted_difficulty=table["predicted_difficulty"].to_numpy(),
    )

    assert result.global_step == 1
    assert len(table) == 2
    assert calibrated.shape == (2,)
    assert (tmp_path / "training_state.pt").exists()
