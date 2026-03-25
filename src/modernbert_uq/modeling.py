from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from modernbert_uq.losses import compute_joint_loss


@dataclass
class UQModelOutput:
    logits_cls: torch.Tensor
    difficulty_pred: torch.Tensor
    loss: torch.Tensor | None = None
    pooled_output: torch.Tensor | None = None
    member_logits: torch.Tensor | None = None
    mean_probability: torch.Tensor | None = None
    probability_variance: torch.Tensor | None = None
    predictive_entropy: torch.Tensor | None = None
    mutual_information: torch.Tensor | None = None
    uncertainty_method: str = "none"


class ModernBertUQModel(nn.Module):
    def __init__(
        self,
        encoder: nn.Module | None = None,
        model_name: str | None = None,
        hidden_size: int | None = None,
        dropout_prob: float = 0.1,
        cls_loss_weight: float = 1.0,
        difficulty_loss_weight: float = 1.0,
        pooling_strategy: str = "mean",
        uq_method: str = "none",
        num_ensemble_members: int = 1,
    ):
        super().__init__()
        if encoder is None:
            if model_name is None:
                raise ValueError("model_name is required when encoder is not provided")
            from transformers import AutoModel

            encoder = AutoModel.from_pretrained(model_name)

        inferred_hidden_size = hidden_size or getattr(encoder.config, "hidden_size", None)
        if inferred_hidden_size is None:
            raise ValueError("hidden_size could not be inferred from encoder config")

        self.encoder = encoder
        self.dropout = nn.Dropout(dropout_prob)
        self.pooling_strategy = pooling_strategy
        self.uq_method = uq_method
        self.num_ensemble_members = num_ensemble_members
        if uq_method == "head_ensemble":
            self.classification_head = nn.ModuleList(
                [nn.Linear(inferred_hidden_size, 1) for _ in range(num_ensemble_members)]
            )
        else:
            self.classification_head = nn.Linear(inferred_hidden_size, 1)
        self.difficulty_head = nn.Linear(inferred_hidden_size, 1)
        self.cls_loss_weight = cls_loss_weight
        self.difficulty_loss_weight = difficulty_loss_weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        difficulty: torch.Tensor | None = None,
        label_mask: torch.Tensor | None = None,
        difficulty_mask: torch.Tensor | None = None,
        **_: Any,
    ) -> UQModelOutput:
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = pool_hidden_states(
            encoder_outputs.last_hidden_state,
            attention_mask=attention_mask,
            pooling_strategy=self.pooling_strategy,
        )
        pooled = self.dropout(pooled)

        member_logits = None
        mean_probability = None
        probability_variance = None
        predictive_entropy = None
        mutual_information = None

        if self.uq_method == "head_ensemble":
            member_logits = torch.stack(
                [head(pooled).squeeze(-1) for head in self.classification_head],
                dim=1,
            )
            logits_cls = member_logits.mean(dim=1)
            probability_samples = torch.sigmoid(member_logits)
            mean_probability = probability_samples.mean(dim=1)
            probability_variance = probability_samples.var(dim=1, unbiased=False)
            sample_entropy = binary_entropy(probability_samples)
            predictive_entropy = binary_entropy(mean_probability)
            mutual_information = predictive_entropy - sample_entropy.mean(dim=1)
        else:
            logits_cls = self.classification_head(pooled).squeeze(-1)
        difficulty_pred = torch.sigmoid(self.difficulty_head(pooled).squeeze(-1))
        if self.uq_method == "head_ensemble":
            cls_loss = compute_ensemble_classification_loss(
                member_logits=member_logits,
                labels=labels,
                label_mask=label_mask,
            )
            difficulty_loss = compute_joint_loss(
                logits_cls=logits_cls,
                difficulty_pred=difficulty_pred,
                labels=None,
                difficulty_targets=difficulty,
                label_mask=None,
                difficulty_mask=difficulty_mask,
                cls_weight=0.0,
                difficulty_weight=self.difficulty_loss_weight,
            )
            loss = None
            if cls_loss is not None:
                loss = self.cls_loss_weight * cls_loss
            if difficulty_loss is not None:
                loss = difficulty_loss if loss is None else loss + difficulty_loss
        else:
            loss = compute_joint_loss(
                logits_cls=logits_cls,
                difficulty_pred=difficulty_pred,
                labels=labels,
                difficulty_targets=difficulty,
                label_mask=label_mask,
                difficulty_mask=difficulty_mask,
                cls_weight=self.cls_loss_weight,
                difficulty_weight=self.difficulty_loss_weight,
            )
        return UQModelOutput(
            logits_cls=logits_cls,
            difficulty_pred=difficulty_pred,
            loss=loss,
            pooled_output=pooled,
            member_logits=member_logits,
            mean_probability=mean_probability,
            probability_variance=probability_variance,
            predictive_entropy=predictive_entropy,
            mutual_information=mutual_information,
            uncertainty_method=self.uq_method,
        )


def pool_hidden_states(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    pooling_strategy: str = "mean",
) -> torch.Tensor:
    if pooling_strategy == "mean":
        return masked_mean_pool(hidden_states, attention_mask=attention_mask)
    if pooling_strategy == "last":
        return last_token_pool(hidden_states, attention_mask=attention_mask)
    raise ValueError(f"Unsupported pooling_strategy: {pooling_strategy}")


def masked_mean_pool(
    hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None
) -> torch.Tensor:
    if attention_mask is None:
        return hidden_states.mean(dim=1)

    mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
    masked_hidden = hidden_states * mask
    denom = mask.sum(dim=1).clamp_min(1.0)
    return masked_hidden.sum(dim=1) / denom


def last_token_pool(
    hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None
) -> torch.Tensor:
    if attention_mask is None:
        return hidden_states[:, -1]
    last_indices = attention_mask.shape[1] - 1 - attention_mask.flip(dims=[1]).argmax(dim=1)
    return hidden_states[torch.arange(hidden_states.shape[0], device=hidden_states.device), last_indices]


def compute_ensemble_classification_loss(
    member_logits: torch.Tensor | None,
    labels: torch.Tensor | None,
    label_mask: torch.Tensor | None,
) -> torch.Tensor | None:
    if member_logits is None or labels is None:
        return None
    if label_mask is None:
        label_mask = torch.ones_like(labels, dtype=torch.bool)
    label_mask = label_mask.to(device=member_logits.device, dtype=torch.bool)
    if not torch.any(label_mask):
        return None
    masked_logits = member_logits[label_mask]
    masked_labels = labels.to(device=member_logits.device, dtype=member_logits.dtype)[label_mask]
    expanded_labels = masked_labels.unsqueeze(1).expand_as(masked_logits)
    return F.binary_cross_entropy_with_logits(masked_logits, expanded_labels)


def binary_entropy(probabilities: torch.Tensor) -> torch.Tensor:
    clipped = probabilities.clamp(min=1e-6, max=1.0 - 1e-6)
    return -(clipped * torch.log(clipped) + (1.0 - clipped) * torch.log(1.0 - clipped))
