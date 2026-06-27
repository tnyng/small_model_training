import numpy as np
import pytest


@pytest.mark.requires_torch
def test_corn_loss_handles_all_label_positions():
    import torch

    from modernbert_uq.ordinal.losses import compute_corn_loss

    logits = torch.randn(6, 8, requires_grad=True)
    labels = torch.tensor([0, 1, 3, 7, 8, 4], dtype=torch.long)
    mask = torch.ones(6, dtype=torch.bool)

    loss = compute_corn_loss(
        threshold_logits=logits,
        class_indices=labels,
        label_mask=mask,
    )
    assert loss is not None
    assert loss.ndim == 0
    loss.backward()
    assert logits.grad is not None


@pytest.mark.requires_torch
def test_corn_loss_masked_out_samples_do_not_contribute():
    import torch
    import torch.nn.functional as F

    from modernbert_uq.ordinal.losses import compute_corn_loss

    torch.manual_seed(0)
    logits_full = torch.randn(4, 3, requires_grad=False)
    labels_full = torch.tensor([0, 2, 1, 3], dtype=torch.long)
    keep_mask = torch.tensor([True, True, False, True])

    loss_masked = compute_corn_loss(
        threshold_logits=logits_full,
        class_indices=labels_full,
        label_mask=keep_mask,
    )

    logits_sub = logits_full[keep_mask]
    labels_sub = labels_full[keep_mask]
    loss_sub = compute_corn_loss(
        threshold_logits=logits_sub,
        class_indices=labels_sub,
    )
    assert torch.allclose(loss_masked, loss_sub, atol=1e-6)


@pytest.mark.requires_torch
def test_corn_loss_returns_none_when_nothing_valid():
    import torch

    from modernbert_uq.ordinal.losses import compute_corn_loss

    logits = torch.randn(3, 4)
    labels = torch.tensor([0, 1, 2], dtype=torch.long)
    mask = torch.zeros(3, dtype=torch.bool)

    assert compute_corn_loss(logits, labels, mask) is None


@pytest.mark.requires_torch
def test_ordinal_model_forward_produces_valid_distribution_and_monotone_cumulative():
    from types import SimpleNamespace

    import torch
    import torch.nn as nn

    from modernbert_uq.ordinal.modeling import ModernBertOrdinalModel

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

    model = ModernBertOrdinalModel(
        encoder=DummyEncoder(hidden_size=8),
        num_score_classes=9,
        label_offset=1,
        dropout_prob=0.0,
    )

    batch = {
        "input_ids": torch.tensor([[1, 2, 0], [3, 4, 5]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.long),
        "labels": torch.tensor([0, 8], dtype=torch.long),
        "difficulty": torch.tensor([0.2, 0.8]),
        "label_mask": torch.tensor([True, True]),
        "difficulty_mask": torch.tensor([True, True]),
    }
    outputs = model(**batch)

    assert outputs.threshold_logits.shape == (2, 8)
    assert outputs.cumulative_gt.shape == (2, 8)
    assert torch.all(outputs.cumulative_gt >= 0.0)
    assert torch.all(outputs.cumulative_gt <= 1.0)
    diffs = torch.diff(outputs.cumulative_gt, dim=-1)
    assert torch.all(diffs <= 1e-6)

    assert outputs.class_probabilities.shape == (2, 9)
    assert torch.allclose(
        outputs.class_probabilities.sum(dim=-1),
        torch.ones(2),
        atol=1e-5,
    )
    assert torch.all(outputs.class_probabilities >= 0)
    assert ((outputs.expected_score >= 1.0) & (outputs.expected_score <= 9.0)).all()
    assert outputs.loss is not None
    outputs.loss.backward()


def test_ordinal_evaluate_flags_monotonicity_violations():
    from modernbert_uq.ordinal.evaluate import evaluate_ordinal_predictions

    class_probabilities = np.array(
        [[0.1, 0.2, 0.3, 0.4], [0.25, 0.25, 0.25, 0.25]], dtype=float
    )
    cumulative_gt = np.array(
        [[0.9, 0.6, 0.3], [0.4, 0.5, 0.2]],
        dtype=float,
    )
    expected_scores = (class_probabilities * np.array([1, 2, 3, 4])).sum(axis=-1)
    argmax_scores = class_probabilities.argmax(axis=-1).astype(float) + 1.0
    threshold_scores = (cumulative_gt > 0.5).sum(axis=-1).astype(float) + 1.0
    gold_scores = np.array([4.0, 2.0])

    metrics = evaluate_ordinal_predictions(
        gold_scores=gold_scores,
        class_probabilities=class_probabilities,
        cumulative_gt=cumulative_gt,
        expected_scores=expected_scores,
        argmax_scores=argmax_scores,
        threshold_scores=threshold_scores,
        label_offset=1,
    )
    assert metrics["score_ordinal"]["monotonic_violation_count"] == 1.0
    assert "quadratic_weighted_kappa" in metrics["score_classification"]
    assert "nll" in metrics["score_distribution"]


@pytest.mark.requires_torch
def test_mc_dropout_distribution_statistics_on_dummy_model():
    from types import SimpleNamespace

    import torch
    import torch.nn as nn

    from modernbert_uq.distribution.modeling import ModernBertDistributionModel
    from modernbert_uq.distribution.uncertainty import mc_dropout_distribution_predict

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

    model = ModernBertDistributionModel(
        encoder=DummyEncoder(hidden_size=8),
        num_score_classes=9,
        label_offset=1,
        dropout_prob=0.5,
    )

    batch = {
        "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1]], dtype=torch.long),
    }
    bin_values = torch.arange(1, 10, dtype=torch.float32)
    outputs = mc_dropout_distribution_predict(
        model=model,
        batch=batch,
        num_passes=5,
        bin_values=bin_values,
    )

    assert outputs["probability_samples"].shape == (5, 2, 9)
    assert outputs["mean_probability"].shape == (2, 9)
    assert torch.allclose(
        outputs["mean_probability"].sum(dim=-1),
        torch.ones(2),
        atol=1e-5,
    )
    assert outputs["predictive_entropy"].shape == (2,)
    assert outputs["expected_entropy"].shape == (2,)
    assert torch.all(outputs["mutual_information"] >= -1e-5)
    assert outputs["expected_score_variance"].shape == (2,)
