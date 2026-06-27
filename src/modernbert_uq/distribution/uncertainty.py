from __future__ import annotations

import torch
from torch import nn

from modernbert_uq.uncertainty import enable_dropout


def mc_dropout_distribution_predict(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    num_passes: int,
    bin_values: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Run `num_passes` stochastic forward passes with dropout enabled.

    Returns per-batch aggregate uncertainty statistics decomposed into:
    - predictive entropy   H[mean_p]    (total uncertainty)
    - expected entropy     mean H[p_t]  (aleatoric component)
    - mutual information   H[mean_p] - mean H[p_t]  (epistemic component)
    """
    was_training = model.training
    model.eval()
    enable_dropout(model)

    probability_samples: list[torch.Tensor] = []
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
            expected_score_samples.append(outputs.expected_score)
            if outputs.difficulty_pred is not None:
                has_difficulty = True
                difficulty_samples.append(outputs.difficulty_pred)

    if was_training:
        model.train()
    else:
        model.eval()

    probability_tensor = torch.stack(probability_samples, dim=0)       # (T, B, K)
    expected_score_tensor = torch.stack(expected_score_samples, dim=0)  # (T, B)

    mean_probability = probability_tensor.mean(dim=0)                  # (B, K)
    probability_std = probability_tensor.std(dim=0, unbiased=False)    # (B, K)

    predictive_entropy = _categorical_entropy(mean_probability)        # (B,)
    per_sample_entropy = _categorical_entropy(probability_tensor)      # (T, B)
    expected_entropy = per_sample_entropy.mean(dim=0)                  # (B,)
    mutual_information = predictive_entropy - expected_entropy         # (B,)

    mean_expected_score = expected_score_tensor.mean(dim=0)            # (B,)
    expected_score_variance = expected_score_tensor.var(dim=0, unbiased=False)

    bin_values = bin_values.to(device=mean_probability.device, dtype=mean_probability.dtype)
    argmax_score = mean_probability.argmax(dim=-1).to(bin_values.dtype) + float(bin_values[0].item())

    result: dict[str, torch.Tensor] = {
        "probability_samples": probability_tensor,
        "mean_probability": mean_probability,
        "probability_std": probability_std,
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
