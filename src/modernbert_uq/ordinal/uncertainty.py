from __future__ import annotations

import torch
from torch import nn

from modernbert_uq.uncertainty import enable_dropout


def mc_dropout_ordinal_predict(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    num_passes: int,
    bin_values: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """MC-dropout wrapper for the CORN head.

    Uses `class_probabilities` from each stochastic pass (derived from the
    cumulative conditional probabilities). Aggregates the same way as the
    distribution subpackage so downstream tooling can treat the outputs
    uniformly.
    """
    was_training = model.training
    model.eval()
    enable_dropout(model)

    probability_samples: list[torch.Tensor] = []
    cumulative_samples: list[torch.Tensor] = []
    expected_score_samples: list[torch.Tensor] = []
    difficulty_samples: list[torch.Tensor] = []
    has_difficulty = False

    with torch.no_grad():
        for _ in range(num_passes):
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
            )
            probability_samples.append(outputs.class_probabilities)
            cumulative_samples.append(outputs.cumulative_gt)
            expected_score_samples.append(outputs.expected_score)
            if outputs.difficulty_pred is not None:
                has_difficulty = True
                difficulty_samples.append(outputs.difficulty_pred)

    if was_training:
        model.train()
    else:
        model.eval()

    probability_tensor = torch.stack(probability_samples, dim=0)        # (T, B, K)
    cumulative_tensor = torch.stack(cumulative_samples, dim=0)          # (T, B, K-1)
    expected_score_tensor = torch.stack(expected_score_samples, dim=0)  # (T, B)

    mean_probability = probability_tensor.mean(dim=0)
    probability_std = probability_tensor.std(dim=0, unbiased=False)
    mean_cumulative = cumulative_tensor.mean(dim=0)
    cumulative_std = cumulative_tensor.std(dim=0, unbiased=False)

    predictive_entropy = _categorical_entropy(mean_probability)
    per_sample_entropy = _categorical_entropy(probability_tensor)
    expected_entropy = per_sample_entropy.mean(dim=0)
    mutual_information = predictive_entropy - expected_entropy

    mean_expected_score = expected_score_tensor.mean(dim=0)
    expected_score_variance = expected_score_tensor.var(dim=0, unbiased=False)

    bin_values = bin_values.to(device=mean_probability.device, dtype=mean_probability.dtype)
    argmax_score = mean_probability.argmax(dim=-1).to(bin_values.dtype) + float(bin_values[0].item())

    result: dict[str, torch.Tensor] = {
        "probability_samples": probability_tensor,
        "cumulative_samples": cumulative_tensor,
        "mean_probability": mean_probability,
        "probability_std": probability_std,
        "mean_cumulative_gt": mean_cumulative,
        "cumulative_gt_std": cumulative_std,
        "predictive_entropy": predictive_entropy,
        "expected_entropy": expected_entropy,
        "mutual_information": mutual_information,
        "expected_score_samples": expected_score_tensor,
        "mean_expected_score": mean_expected_score,
        "expected_score_variance": expected_score_variance,
        "argmax_score_from_mean": argmax_score,
    }
    if has_difficulty:
        difficulty_tensor = torch.stack(difficulty_samples, dim=0)
        result["difficulty_samples"] = difficulty_tensor
        result["mean_difficulty"] = difficulty_tensor.mean(dim=0)
        result["difficulty_variance"] = difficulty_tensor.var(dim=0, unbiased=False)
    return result


def _categorical_entropy(probabilities: torch.Tensor) -> torch.Tensor:
    clipped = probabilities.clamp_min(1e-12)
    return -(clipped * torch.log(clipped)).sum(dim=-1)
