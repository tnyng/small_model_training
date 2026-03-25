import pytest

pytestmark = pytest.mark.requires_torch


def test_run_training_saves_training_state(tmp_path):
    from types import SimpleNamespace

    import torch
    import torch.nn as nn
    import accelerate  # noqa: F401

    from modernbert_uq.modeling import ModernBertUQModel
    from modernbert_uq.train import TrainConfig, run_training

    class DummyEncoder(nn.Module):
        def __init__(self, hidden_size: int):
            super().__init__()
            self.config = SimpleNamespace(hidden_size=hidden_size)
            self.embedding = nn.Embedding(64, hidden_size)

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

    model = ModernBertUQModel(encoder=DummyEncoder(hidden_size=12), dropout_prob=0.1)
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

    config = TrainConfig(
        learning_rate=1e-3,
        num_epochs=1,
        gradient_accumulation_steps=1,
        output_dir=tmp_path,
    )

    result = run_training(model, train_batches, eval_batches, config)

    assert result.global_step == 1
    assert result.best_eval_loss is not None
    assert (tmp_path / "training_state.pt").exists()


def test_predict_batches_exposes_head_ensemble_uncertainty_fields(tmp_path):
    from types import SimpleNamespace

    import torch
    import torch.nn as nn
    import accelerate  # noqa: F401

    from modernbert_uq.modeling import ModernBertUQModel
    from modernbert_uq.train import TrainConfig, predict_batches, run_training

    class DummyEncoder(nn.Module):
        def __init__(self, hidden_size: int):
            super().__init__()
            self.config = SimpleNamespace(hidden_size=hidden_size)
            self.embedding = nn.Embedding(64, hidden_size)

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

    model = ModernBertUQModel(
        encoder=DummyEncoder(hidden_size=12),
        dropout_prob=0.0,
        uq_method="head_ensemble",
        num_ensemble_members=3,
    )
    train_batches = [
        build_batch(
            torch.tensor([[1, 2, 3], [4, 5, 0]], dtype=torch.long),
            torch.tensor([1.0, 0.0]),
            torch.tensor([0.2, 0.7]),
        )
    ]

    run_training(
        model,
        train_batches,
        train_batches,
        TrainConfig(
            learning_rate=1e-3,
            num_epochs=1,
            gradient_accumulation_steps=1,
            output_dir=tmp_path,
        ),
    )

    predictions = predict_batches(model, train_batches)

    assert predictions["uncertainty_method"] == "head_ensemble"
    assert predictions["member_logits"].shape == (2, 3)
    assert predictions["probability_variance"].shape == (2,)
    assert predictions["mutual_information"].shape == (2,)
