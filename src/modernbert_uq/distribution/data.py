from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class DistributionBatchCollator:
    num_score_classes: int
    label_offset: int = 1

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        labels_raw = [feature.get("labels") for feature in features]
        difficulty_raw = [feature.get("difficulty") for feature in features]

        label_mask = [value is not None and pd.notna(value) for value in labels_raw]
        difficulty_mask = [value is not None and pd.notna(value) for value in difficulty_raw]

        indices: list[int] = []
        for value, is_valid in zip(labels_raw, label_mask):
            if not is_valid:
                indices.append(0)
                continue
            class_index = int(round(float(value))) - self.label_offset
            if class_index < 0 or class_index >= self.num_score_classes:
                raise ValueError(
                    f"Label {value!r} maps to class index {class_index} which is "
                    f"outside [0, {self.num_score_classes - 1}] for label_offset="
                    f"{self.label_offset}."
                )
            indices.append(class_index)

        return {
            "input_ids": torch.tensor(
                [feature["input_ids"] for feature in features], dtype=torch.long
            ),
            "attention_mask": torch.tensor(
                [feature["attention_mask"] for feature in features], dtype=torch.long
            ),
            "labels": torch.tensor(indices, dtype=torch.long),
            "difficulty": torch.tensor(
                [
                    float(value) if is_valid else 0.0
                    for value, is_valid in zip(difficulty_raw, difficulty_mask)
                ],
                dtype=torch.float32,
            ),
            "label_mask": torch.tensor(label_mask, dtype=torch.bool),
            "difficulty_mask": torch.tensor(difficulty_mask, dtype=torch.bool),
        }
