from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_corn_loss(
    threshold_logits: torch.Tensor,
    class_indices: torch.Tensor | None,
    label_mask: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """CORN (Shi et al. 2022) conditional training loss.

    threshold_logits: (B, K-1). Each column j scores P(y > j | y > j-1).
    class_indices:    (B,) integer class index in [0, K-1].
    label_mask:       (B,) optional; False entries are ignored.

    For sample i with class index c_i, the sample contributes to the loss for
    every threshold j in [0, c_i] (if c_i < K-1) or [0, K-2] (if c_i == K-1).
    Target for threshold j is 1 when c_i > j, else 0. Loss is averaged over
    (sample, threshold) contributions.
    """
    if class_indices is None:
        return None

    if label_mask is None:
        label_mask = torch.ones(
            threshold_logits.shape[0],
            dtype=torch.bool,
            device=threshold_logits.device,
        )
    else:
        label_mask = label_mask.to(device=threshold_logits.device, dtype=torch.bool)
    if not torch.any(label_mask):
        return None

    num_thresholds = threshold_logits.shape[-1]
    device = threshold_logits.device
    class_indices = class_indices.to(device=device, dtype=torch.long)

    threshold_range = torch.arange(num_thresholds, device=device).unsqueeze(0)  # (1, K-1)
    in_subset = (class_indices.unsqueeze(-1) >= threshold_range) & label_mask.unsqueeze(-1)
    if not torch.any(in_subset):
        return None

    targets = (class_indices.unsqueeze(-1) > threshold_range).to(threshold_logits.dtype)
    per_element = F.binary_cross_entropy_with_logits(
        threshold_logits, targets, reduction="none"
    )
    masked = per_element * in_subset.to(threshold_logits.dtype)
    total_loss = masked.sum()
    total_count = in_subset.sum().clamp_min(1)
    return total_loss / total_count


def compute_ordinal_difficulty_loss(
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
