import pytest

pytestmark = pytest.mark.requires_torch


def test_mc_dropout_predict_returns_stochastic_summary():
    from types import SimpleNamespace

    import torch
    import torch.nn as nn

    from modernbert_uq.modeling import ModernBertUQModel
    from modernbert_uq.uncertainty import mc_dropout_predict

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

    model = ModernBertUQModel(encoder=DummyEncoder(hidden_size=8), dropout_prob=0.5)
    batch = {
        "input_ids": torch.tensor([[1, 2, 0], [3, 4, 5]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.long),
    }

    summary = mc_dropout_predict(model, batch, num_passes=8)

    assert summary["probability_samples"].shape == (8, 2)
    assert summary["difficulty_samples"].shape == (8, 2)
    assert summary["probability_variance"].shape == (2,)
    assert summary["difficulty_variance"].shape == (2,)
    assert torch.any(summary["probability_samples"][0] != summary["probability_samples"][1])
    assert torch.all(summary["predictive_entropy"] >= 0.0)
