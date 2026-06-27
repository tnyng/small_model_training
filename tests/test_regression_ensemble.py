from __future__ import annotations

import pytest

pytest.importorskip("torch")

import torch
import torch.nn as nn

from modernbert_uq.modeling import ModernBertUQModel


def _make_tiny_encoder(hidden_size: int = 16) -> nn.Module:
    class TinyEncoder(nn.Module):
        class Config:
            hidden_size = 16
        def __init__(self):
            super().__init__()
            self.config = TinyEncoder.Config()
        def forward(self, input_ids, attention_mask=None):
            batch, seq = input_ids.shape
            hidden = torch.zeros(batch, seq, hidden_size)
            class Out:
                last_hidden_state = hidden
            return Out()
    return TinyEncoder()


@pytest.mark.requires_torch
def test_regression_ensemble_forward_shape():
    encoder = _make_tiny_encoder(hidden_size=16)
    model = ModernBertUQModel(
        encoder=encoder,
        hidden_size=16,
        score_task_type="regression",
        uq_method="head_ensemble",
        num_ensemble_members=3,
    )
    input_ids = torch.zeros(2, 8, dtype=torch.long)
    out = model(input_ids=input_ids)
    assert out.score_pred.shape == (2,)
    assert out.member_scores is not None
    assert out.member_scores.shape == (2, 3)


@pytest.mark.requires_torch
def test_regression_ensemble_member_scores_in_range():
    encoder = _make_tiny_encoder(hidden_size=16)
    model = ModernBertUQModel(
        encoder=encoder,
        hidden_size=16,
        score_task_type="regression",
        uq_method="head_ensemble",
        num_ensemble_members=3,
    )
    input_ids = torch.zeros(4, 8, dtype=torch.long)
    out = model(input_ids=input_ids)
    assert (out.member_scores > 0.0).all()
    assert (out.member_scores < 1.0).all()


@pytest.mark.requires_torch
def test_regression_ensemble_loss_computed():
    encoder = _make_tiny_encoder(hidden_size=16)
    model = ModernBertUQModel(
        encoder=encoder,
        hidden_size=16,
        score_task_type="regression",
        uq_method="head_ensemble",
        num_ensemble_members=3,
    )
    input_ids = torch.zeros(2, 8, dtype=torch.long)
    labels = torch.tensor([0.2, 0.8])
    label_mask = torch.tensor([True, True])
    out = model(input_ids=input_ids, labels=labels, label_mask=label_mask)
    assert out.loss is not None
    assert out.loss.shape == ()
