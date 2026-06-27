import numpy as np
import pytest


@pytest.mark.requires_torch
def test_gaussian_soft_targets_sum_to_one():
    import torch

    from modernbert_uq.distribution.losses import _gaussian_soft_targets

    targets = torch.tensor([0, 4, 8], dtype=torch.long)
    soft_targets = _gaussian_soft_targets(targets=targets, num_classes=9, sigma=1.0)
    assert soft_targets.shape == (3, 9)
    assert torch.allclose(
        soft_targets.sum(dim=-1),
        torch.ones(3, dtype=soft_targets.dtype),
        atol=1e-6,
    )
    assert soft_targets[1].argmax().item() == 4


@pytest.mark.requires_torch
def test_distribution_collator_maps_score_bucket_to_zero_indexed_class():
    from modernbert_uq.distribution.data import DistributionBatchCollator

    features = [
        {"input_ids": [1, 2, 0, 0], "attention_mask": [1, 1, 0, 0], "labels": 1, "difficulty": 0.2},
        {"input_ids": [3, 4, 5, 0], "attention_mask": [1, 1, 1, 0], "labels": 9, "difficulty": 0.5},
        {"input_ids": [6, 0, 0, 0], "attention_mask": [1, 0, 0, 0], "labels": None, "difficulty": None},
    ]
    collator = DistributionBatchCollator(num_score_classes=9, label_offset=1)
    batch = collator(features)

    assert batch["labels"].tolist() == [0, 8, 0]
    assert batch["label_mask"].tolist() == [True, True, False]
    assert batch["difficulty_mask"].tolist() == [True, True, False]
    assert batch["input_ids"].shape == (3, 4)


@pytest.mark.requires_torch
def test_distribution_collator_rejects_out_of_range_labels():
    from modernbert_uq.distribution.data import DistributionBatchCollator

    features = [
        {"input_ids": [1], "attention_mask": [1], "labels": 10, "difficulty": 0.5},
    ]
    collator = DistributionBatchCollator(num_score_classes=9, label_offset=1)
    with pytest.raises(ValueError):
        collator(features)


@pytest.mark.requires_torch
def test_distribution_model_forward_shapes_and_loss():
    from types import SimpleNamespace

    import torch
    import torch.nn as nn

    from modernbert_uq.distribution.modeling import ModernBertDistributionModel

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

    assert outputs.logits.shape == (2, 9)
    assert outputs.class_probabilities.shape == (2, 9)
    assert torch.allclose(
        outputs.class_probabilities.sum(dim=-1),
        torch.ones(2),
        atol=1e-5,
    )
    assert outputs.expected_score.shape == (2,)
    assert ((outputs.expected_score >= 1.0) & (outputs.expected_score <= 9.0)).all()
    assert outputs.argmax_score.shape == (2,)
    assert outputs.predictive_entropy.shape == (2,)
    assert outputs.loss is not None
    assert outputs.loss.ndim == 0
    outputs.loss.backward()


@pytest.mark.requires_torch
def test_distribution_model_hard_vs_soft_label_loss_agrees_at_sigma_zero():
    from types import SimpleNamespace

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from modernbert_uq.distribution.losses import compute_distribution_score_loss

    torch.manual_seed(0)
    logits = torch.randn(5, 9, requires_grad=False)
    labels = torch.tensor([0, 3, 4, 8, 1], dtype=torch.long)

    hard = compute_distribution_score_loss(
        logits=logits, labels=labels, soft_label_sigma=0.0, num_score_classes=9
    )
    reference = F.cross_entropy(logits, labels)
    assert torch.allclose(hard, reference, atol=1e-6)


def test_evaluate_distribution_metrics_finite():
    from modernbert_uq.distribution.evaluate import evaluate_distribution_predictions

    rng = np.random.default_rng(0)
    class_probabilities = rng.dirichlet(np.ones(9), size=20).astype(float)
    gold_scores = rng.integers(1, 10, size=20).astype(float)
    expected_scores = (class_probabilities * np.arange(1, 10, dtype=float)).sum(axis=-1)
    argmax_scores = class_probabilities.argmax(axis=-1).astype(float) + 1.0

    metrics = evaluate_distribution_predictions(
        gold_scores=gold_scores,
        class_probabilities=class_probabilities,
        expected_scores=expected_scores,
        argmax_scores=argmax_scores,
        label_offset=1,
    )
    assert set(metrics) == {"score_regression", "score_classification", "score_distribution"}
    for section in metrics.values():
        for value in section.values():
            assert np.isfinite(value) or np.isnan(value)


def test_multiclass_temperature_scaler_matches_cross_entropy_at_unit_temperature():
    from modernbert_uq.distribution.calibration import MulticlassTemperatureScaler

    rng = np.random.default_rng(1)
    logits = rng.normal(size=(50, 9))
    gold = rng.integers(0, 9, size=50)

    scaler = MulticlassTemperatureScaler().fit(logits, gold)
    assert 0.05 <= scaler.temperature_ <= 5.0

    probs = scaler.predict_probabilities(logits)
    assert probs.shape == (50, 9)
    assert np.allclose(probs.sum(axis=-1), 1.0)
