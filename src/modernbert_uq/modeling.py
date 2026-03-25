from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from modernbert_uq.losses import compute_joint_loss


@dataclass
class UQModelOutput:
    logits_cls: torch.Tensor
    difficulty_pred: torch.Tensor
    loss: torch.Tensor | None = None
    pooled_output: torch.Tensor | None = None


class ModernBertUQModel(nn.Module):
    def __init__(
        self,
        encoder: nn.Module | None = None,
        model_name: str | None = None,
        hidden_size: int | None = None,
        dropout_prob: float = 0.1,
        cls_loss_weight: float = 1.0,
        difficulty_loss_weight: float = 1.0,
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
        pooled = masked_mean_pool(
            encoder_outputs.last_hidden_state, attention_mask=attention_mask
        )
        pooled = self.dropout(pooled)

        logits_cls = self.classification_head(pooled).squeeze(-1)
        difficulty_pred = torch.sigmoid(self.difficulty_head(pooled).squeeze(-1))
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
        )


def masked_mean_pool(
    hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None
) -> torch.Tensor:
    if attention_mask is None:
        return hidden_states.mean(dim=1)

    mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
    masked_hidden = hidden_states * mask
    denom = mask.sum(dim=1).clamp_min(1.0)
    return masked_hidden.sum(dim=1) / denom
