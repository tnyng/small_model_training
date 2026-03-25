from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def compute_joint_loss(
    logits_cls: torch.Tensor,
    difficulty_pred: torch.Tensor,
    labels: torch.Tensor | None = None,
    difficulty_targets: torch.Tensor | None = None,
    label_mask: torch.Tensor | None = None,
    difficulty_mask: torch.Tensor | None = None,
    cls_weight: float = 1.0,
    difficulty_weight: float = 1.0,
) -> torch.Tensor | None:
    cls_loss = _masked_loss(
        F.binary_cross_entropy_with_logits,
        predictions=logits_cls,
        targets=labels,
        mask=label_mask,
    )
    difficulty_loss = _masked_loss(
        F.mse_loss,
        predictions=difficulty_pred,
        targets=difficulty_targets,
        mask=difficulty_mask,
    )

    if cls_loss is None and difficulty_loss is None:
        return None

    total = torch.zeros((), device=logits_cls.device, dtype=logits_cls.dtype)
    if cls_loss is not None:
        total = total + (cls_weight * cls_loss)
    if difficulty_loss is not None:
        total = total + (difficulty_weight * difficulty_loss)
    return total


def _masked_loss(
    loss_fn: Any,
    predictions: torch.Tensor,
    targets: torch.Tensor | None,
    mask: torch.Tensor | None,
) -> torch.Tensor | None:
    if targets is None:
        return None

    if mask is None:
        mask = torch.ones_like(predictions, dtype=torch.bool)
    else:
        mask = mask.to(dtype=torch.bool, device=predictions.device)

    if mask.numel() == 0 or not torch.any(mask):
        return None

    filtered_predictions = predictions[mask]
    filtered_targets = targets.to(device=predictions.device, dtype=predictions.dtype)[mask]
    return loss_fn(filtered_predictions, filtered_targets)
