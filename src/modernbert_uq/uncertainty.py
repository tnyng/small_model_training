from __future__ import annotations

import torch
from torch import nn


def mc_dropout_predict(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    num_passes: int,
) -> dict[str, torch.Tensor]:
    was_training = model.training
    model.eval()
    enable_dropout(model)

    probability_samples = []
    difficulty_samples = []
    with torch.no_grad():
        for _ in range(num_passes):
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
            )
            probability_samples.append(torch.sigmoid(outputs.logits_cls))
            difficulty_samples.append(outputs.difficulty_pred)

    if was_training:
        model.train()
    else:
        model.eval()

    probability_tensor = torch.stack(probability_samples, dim=0)
    difficulty_tensor = torch.stack(difficulty_samples, dim=0)
    mean_probability = probability_tensor.mean(dim=0)
    sample_entropy = binary_entropy(probability_tensor)
    predictive_entropy = binary_entropy(mean_probability)
    mutual_information = predictive_entropy - sample_entropy.mean(dim=0)

    return {
        "probability_samples": probability_tensor,
        "difficulty_samples": difficulty_tensor,
        "mean_probability": mean_probability,
        "probability_variance": probability_tensor.var(dim=0, unbiased=False),
        "predictive_entropy": predictive_entropy,
        "mutual_information": mutual_information,
        "mean_difficulty": difficulty_tensor.mean(dim=0),
        "difficulty_variance": difficulty_tensor.var(dim=0, unbiased=False),
    }


def enable_dropout(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


def binary_entropy(probabilities: torch.Tensor) -> torch.Tensor:
    clipped = probabilities.clamp(min=1e-6, max=1.0 - 1e-6)
    return -(clipped * torch.log(clipped) + (1.0 - clipped) * torch.log(1.0 - clipped))
