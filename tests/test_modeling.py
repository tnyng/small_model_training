import pytest

pytestmark = pytest.mark.requires_torch


def test_dual_head_model_returns_expected_shapes_and_scalar_loss():
    from types import SimpleNamespace

    import torch
    import torch.nn as nn

    from modernbert_uq.modeling import ModernBertUQModel

    class DummyEncoder(nn.Module):
        def __init__(self, hidden_size: int):
            super().__init__()
            self.config = SimpleNamespace(hidden_size=hidden_size)
            self.embedding = nn.Embedding(32, hidden_size)

        def forward(self, input_ids, attention_mask=None):
            hidden = self.embedding(input_ids)
            if attention_mask is not None:
                hidden = hidden * attention_mask.unsqueeze(-1)
            return SimpleNamespace(last_hidden_state=hidden)

    model = ModernBertUQModel(encoder=DummyEncoder(hidden_size=8), dropout_prob=0.4)

    batch = {
        "input_ids": torch.tensor([[1, 2, 0], [3, 4, 5]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.long),
        "labels": torch.tensor([1.0, 0.0]),
        "difficulty": torch.tensor([0.2, 0.8]),
        "label_mask": torch.tensor([True, True]),
        "difficulty_mask": torch.tensor([True, True]),
    }

    outputs = model(**batch)

    assert outputs.logits_cls.shape == (2,)
    assert outputs.difficulty_pred.shape == (2,)
    assert outputs.loss is not None
    assert outputs.loss.ndim == 0


def test_head_ensemble_returns_member_predictions_and_uncertainty_stats():
    from types import SimpleNamespace

    import torch
    import torch.nn as nn

    from modernbert_uq.modeling import ModernBertUQModel

    class DummyEncoder(nn.Module):
        def __init__(self, hidden_size: int):
            super().__init__()
            self.config = SimpleNamespace(hidden_size=hidden_size)
            self.embedding = nn.Embedding(32, hidden_size)

        def forward(self, input_ids, attention_mask=None):
            hidden = self.embedding(input_ids)
            if attention_mask is not None:
                hidden = hidden * attention_mask.unsqueeze(-1)
            return SimpleNamespace(last_hidden_state=hidden)

    model = ModernBertUQModel(
        encoder=DummyEncoder(hidden_size=8),
        dropout_prob=0.0,
        uq_method="head_ensemble",
        num_ensemble_members=3,
    )

    batch = {
        "input_ids": torch.tensor([[1, 2, 0], [3, 4, 5]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.long),
        "labels": torch.tensor([1.0, 0.0]),
        "difficulty": torch.tensor([0.2, 0.8]),
        "label_mask": torch.tensor([True, True]),
        "difficulty_mask": torch.tensor([True, True]),
    }

    outputs = model(**batch)

    assert outputs.logits_cls.shape == (2,)
    assert outputs.member_logits.shape == (2, 3)
    assert outputs.mean_probability.shape == (2,)
    assert outputs.probability_variance.shape == (2,)
    assert outputs.predictive_entropy.shape == (2,)
    assert outputs.mutual_information.shape == (2,)
