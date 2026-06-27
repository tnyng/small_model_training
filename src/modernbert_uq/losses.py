from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def bradley_terry_loss(
    score_a: torch.Tensor,
    score_b: torch.Tensor,
    margin: torch.Tensor | None = None,
    center_rewards_coeff: float = 0.0,
) -> torch.Tensor:
    """Bradley-Terry pairwise ranking loss.

    score_a is the predicted reward for the preferred (higher-scored) text,
    score_b is the predicted reward for the rejected (lower-scored) text.
    """
    margin_term = 0.0
    if margin is not None:
        margin_term = margin.to(device=score_a.device, dtype=score_a.dtype)
    loss = -F.logsigmoid(score_a - score_b - margin_term).mean()
    if center_rewards_coeff > 0.0:
        loss = loss + center_rewards_coeff * ((score_a + score_b) ** 2).mean()
    return loss


def compute_joint_loss(
    logits_cls: torch.Tensor,
    score_pred: torch.Tensor,
    difficulty_pred: torch.Tensor,
    labels: torch.Tensor | None = None,
    difficulty_targets: torch.Tensor | None = None,
    label_mask: torch.Tensor | None = None,
    difficulty_mask: torch.Tensor | None = None,
    score_task_type: str = "classification",
    cls_weight: float = 1.0,
    difficulty_weight: float = 1.0,
    positive_class_weight: float = 1.0,
    negative_class_weight: float = 1.0,
) -> torch.Tensor | None:
    if score_task_type == "classification":
        cls_loss = _masked_bce_with_logits_loss(
            logits=logits_cls,
            targets=labels,
            mask=label_mask,
            positive_class_weight=positive_class_weight,
            negative_class_weight=negative_class_weight,
        )
    elif score_task_type == "regression":
        cls_loss = _masked_loss(
            F.mse_loss,
            predictions=score_pred,
            targets=labels,
            mask=label_mask,
        )
    else:
        raise ValueError(f"Unsupported score_task_type: {score_task_type}")
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


def _masked_bce_with_logits_loss(
    logits: torch.Tensor,
    targets: torch.Tensor | None,
    mask: torch.Tensor | None,
    positive_class_weight: float = 1.0,
    negative_class_weight: float = 1.0,
) -> torch.Tensor | None:
    if targets is None:
        return None
    if mask is None:
        mask = torch.ones_like(logits, dtype=torch.bool)
    else:
        mask = mask.to(dtype=torch.bool, device=logits.device)
    if mask.numel() == 0 or not torch.any(mask):
        return None

    filtered_logits = logits[mask]
    filtered_targets = targets.to(device=logits.device, dtype=logits.dtype)[mask]
    losses = F.binary_cross_entropy_with_logits(
        filtered_logits,
        filtered_targets,
        reduction="none",
    )
    weights = torch.where(
        filtered_targets >= 0.5,
        torch.full_like(filtered_targets, float(positive_class_weight)),
        torch.full_like(filtered_targets, float(negative_class_weight)),
    )
    return (losses * weights).mean()


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
