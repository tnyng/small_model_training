from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_distribution_score_loss(
    logits: torch.Tensor,
    labels: torch.Tensor | None,
    label_mask: torch.Tensor | None = None,
    num_score_classes: int | None = None,
    soft_label_sigma: float = 0.0,
) -> torch.Tensor | None:
    """Cross-entropy over K ordered score classes.

    - soft_label_sigma == 0: standard cross-entropy on integer class indices.
    - soft_label_sigma  > 0: Gaussian smoothing over adjacent bins; CE with soft targets.
    """
    if labels is None:
        return None

    if label_mask is None:
        label_mask = torch.ones(logits.shape[0], dtype=torch.bool, device=logits.device)
    else:
        label_mask = label_mask.to(device=logits.device, dtype=torch.bool)

    if not torch.any(label_mask):
        return None

    masked_logits = logits[label_mask]
    masked_labels = labels.to(device=logits.device)[label_mask].to(torch.long)

    if soft_label_sigma <= 0.0:
        return F.cross_entropy(masked_logits, masked_labels)

    if num_score_classes is None:
        num_score_classes = logits.shape[-1]

    soft_targets = _gaussian_soft_targets(
        targets=masked_labels,
        num_classes=num_score_classes,
        sigma=float(soft_label_sigma),
    )
    log_probabilities = F.log_softmax(masked_logits, dim=-1)
    return -(soft_targets * log_probabilities).sum(dim=-1).mean()


def compute_distribution_difficulty_loss(
    difficulty_pred: torch.Tensor | None,
    difficulty_targets: torch.Tensor | None,
    difficulty_mask: torch.Tensor | None = None,
) -> torch.Tensor | None:
    if difficulty_pred is None or difficulty_targets is None:
        return None

    if difficulty_mask is None:
        difficulty_mask = torch.ones_like(difficulty_pred, dtype=torch.bool)
    else:
        difficulty_mask = difficulty_mask.to(device=difficulty_pred.device, dtype=torch.bool)

    if not torch.any(difficulty_mask):
        return None

    predictions = difficulty_pred[difficulty_mask]
    targets = difficulty_targets.to(device=difficulty_pred.device, dtype=difficulty_pred.dtype)[
        difficulty_mask
    ]
    return F.mse_loss(predictions, targets)


def _gaussian_soft_targets(
    targets: torch.Tensor,
    num_classes: int,
    sigma: float,
) -> torch.Tensor:
    device = targets.device
    positions = torch.arange(num_classes, device=device, dtype=torch.float32)
    targets_float = targets.to(torch.float32).unsqueeze(-1)
    squared = (positions.unsqueeze(0) - targets_float) ** 2
    unnormalized = torch.exp(-squared / (2.0 * sigma * sigma))
    return unnormalized / unnormalized.sum(dim=-1, keepdim=True)
