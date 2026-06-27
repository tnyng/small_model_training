from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from modernbert_uq.modeling import pool_hidden_states

from modernbert_uq.ordinal.losses import (
    compute_corn_loss,
    compute_ordinal_difficulty_loss,
)


@dataclass
class OrdinalModelOutput:
    threshold_logits: torch.Tensor              # (B, K-1) raw logits for P(y>j|y>j-1)
    cumulative_gt: torch.Tensor                 # (B, K-1) P(y > j)
    class_probabilities: torch.Tensor           # (B, K) derived P(y = k)
    expected_score: torch.Tensor                # (B,) E[y]
    argmax_score: torch.Tensor                  # (B,) argmax over class_probabilities
    threshold_score: torch.Tensor               # (B,) rank from P(y>j) > 0.5 counts
    predictive_entropy: torch.Tensor            # (B,) H[P(y = .)]
    difficulty_pred: torch.Tensor | None = None
    loss: torch.Tensor | None = None
    pooled_output: torch.Tensor | None = None


class ModernBertOrdinalModel(nn.Module):
    """CORN head on top of a ModernBERT encoder."""

    def __init__(
        self,
        encoder: nn.Module | None = None,
        model_name: str | None = None,
        hidden_size: int | None = None,
        num_score_classes: int = 9,
        label_offset: int = 1,
        dropout_prob: float = 0.1,
        pooling_strategy: str = "mean",
        score_loss_weight: float = 1.0,
        difficulty_loss_weight: float = 0.0,
    ):
        super().__init__()
        if encoder is None:
            if model_name is None:
                raise ValueError("model_name is required when encoder is not provided")
            from transformers import AutoModel

            encoder = AutoModel.from_pretrained(
                model_name,
                attn_implementation="sdpa",
            )

        inferred_hidden_size = hidden_size or getattr(encoder.config, "hidden_size", None)
        if inferred_hidden_size is None:
            raise ValueError("hidden_size could not be inferred from encoder config")

        if num_score_classes < 2:
            raise ValueError("num_score_classes must be >= 2")

        self.encoder = encoder
        self.dropout = nn.Dropout(dropout_prob)
        self.pooling_strategy = pooling_strategy
        self.num_score_classes = num_score_classes
        self.label_offset = label_offset
        self.score_loss_weight = score_loss_weight
        self.difficulty_loss_weight = difficulty_loss_weight

        self.num_thresholds = num_score_classes - 1
        self.threshold_head = nn.Linear(inferred_hidden_size, self.num_thresholds)

        self.has_difficulty_head = difficulty_loss_weight > 0.0
        if self.has_difficulty_head:
            self.difficulty_head = nn.Linear(inferred_hidden_size, 1)
        else:
            self.difficulty_head = None

        bin_values = torch.arange(
            label_offset,
            label_offset + num_score_classes,
            dtype=torch.float32,
        )
        self.register_buffer("bin_values", bin_values, persistent=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        difficulty: torch.Tensor | None = None,
        label_mask: torch.Tensor | None = None,
        difficulty_mask: torch.Tensor | None = None,
        **_: Any,
    ) -> OrdinalModelOutput:
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = pool_hidden_states(
            encoder_outputs.last_hidden_state,
            attention_mask=attention_mask,
            pooling_strategy=self.pooling_strategy,
        )
        pooled = self.dropout(pooled)

        threshold_logits = self.threshold_head(pooled)                     # (B, K-1)
        log_conditional = F.logsigmoid(threshold_logits)                   # log P(y>j|y>j-1)
        log_cumulative_gt = torch.cumsum(log_conditional, dim=-1)          # log P(y > j)
        cumulative_gt = log_cumulative_gt.exp()                            # P(y > j)

        # Derive P(y = k) from cumulative probabilities, enforcing monotonicity via clamp.
        batch_size = cumulative_gt.shape[0]
        ones = torch.ones(batch_size, 1, dtype=cumulative_gt.dtype, device=cumulative_gt.device)
        zeros = torch.zeros_like(ones)
        prob_gt_prev = torch.cat([ones, cumulative_gt], dim=-1)            # (B, K)  P(y > k-1)
        prob_gt_curr = torch.cat([cumulative_gt, zeros], dim=-1)           # (B, K)  P(y > k)
        class_probabilities = (prob_gt_prev - prob_gt_curr).clamp_min(0.0)
        class_probabilities = class_probabilities / class_probabilities.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-12)

        expected_score = (class_probabilities * self.bin_values).sum(dim=-1)
        argmax_score = class_probabilities.argmax(dim=-1).to(torch.float32) + float(self.label_offset)
        threshold_score = (cumulative_gt > 0.5).sum(dim=-1).to(torch.float32) + float(
            self.label_offset
        )
        predictive_entropy = -(class_probabilities.clamp_min(1e-12).log() * class_probabilities).sum(
            dim=-1
        )

        difficulty_pred: torch.Tensor | None = None
        if self.difficulty_head is not None:
            difficulty_pred = torch.sigmoid(self.difficulty_head(pooled).squeeze(-1))

        score_loss = compute_corn_loss(
            threshold_logits=threshold_logits,
            class_indices=labels,
            label_mask=label_mask,
        )
        difficulty_loss = compute_ordinal_difficulty_loss(
            difficulty_pred=difficulty_pred,
            difficulty_targets=difficulty,
            difficulty_mask=difficulty_mask,
        )

        loss: torch.Tensor | None = None
        if score_loss is not None:
            loss = self.score_loss_weight * score_loss
        if difficulty_loss is not None:
            contribution = self.difficulty_loss_weight * difficulty_loss
            loss = contribution if loss is None else loss + contribution

        return OrdinalModelOutput(
            threshold_logits=threshold_logits,
            cumulative_gt=cumulative_gt,
            class_probabilities=class_probabilities,
            expected_score=expected_score,
            argmax_score=argmax_score,
            threshold_score=threshold_score,
            predictive_entropy=predictive_entropy,
            difficulty_pred=difficulty_pred,
            loss=loss,
            pooled_output=pooled,
        )
