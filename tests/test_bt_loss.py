from __future__ import annotations

import pytest

pytest.importorskip("torch")

import torch

from modernbert_uq.losses import bradley_terry_loss


def test_bt_loss_returns_scalar():
    score_a = torch.tensor([1.0, 0.8])
    score_b = torch.tensor([0.0, 0.2])
    loss = bradley_terry_loss(score_a, score_b)
    assert loss.shape == ()


def test_bt_loss_is_positive():
    score_a = torch.tensor([1.0, 0.8])
    score_b = torch.tensor([0.0, 0.2])
    loss = bradley_terry_loss(score_a, score_b)
    assert float(loss) > 0.0


def test_bt_loss_lower_when_preferred_wins_more():
    loss_clear = bradley_terry_loss(torch.tensor([2.0]), torch.tensor([-2.0]))
    loss_ambiguous = bradley_terry_loss(torch.tensor([0.1]), torch.tensor([-0.1]))
    assert float(loss_clear) < float(loss_ambiguous)


def test_bt_loss_center_regularization_increases_loss():
    score_a = torch.tensor([1.0])
    score_b = torch.tensor([0.0])
    loss_no_reg = bradley_terry_loss(score_a, score_b, center_rewards_coeff=0.0)
    loss_with_reg = bradley_terry_loss(score_a, score_b, center_rewards_coeff=1.0)
    assert float(loss_with_reg) > float(loss_no_reg)


def test_bt_loss_with_margin_is_harder_than_plain_bt():
    score_a = torch.tensor([0.8, 0.9])
    score_b = torch.tensor([0.3, 0.4])
    plain = bradley_terry_loss(score_a, score_b)
    margin = bradley_terry_loss(score_a, score_b, margin=torch.tensor([0.2, 0.2]))
    assert float(margin) > float(plain)


def test_bt_loss_gradients_flow():
    score_a = torch.tensor([2.0], requires_grad=True)
    score_b = torch.tensor([-2.0], requires_grad=True)
    loss = bradley_terry_loss(score_a, score_b)
    loss.backward()
    assert score_a.grad is not None
    assert score_b.grad is not None
    # score_a is the preferred item: increasing it lowers the loss, so its gradient is negative.
    # score_b is the rejected item: its gradient is positive (gradient descent then pushes it
    # down, widening the margin and lowering the loss).
    assert float(score_a.grad) < 0.0
    assert float(score_b.grad) > 0.0
