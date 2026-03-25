from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

REQUIRED_COLUMNS = {"text", "label", "difficulty"}


def load_records(path: str | Path) -> list[dict[str, Any]]:
    source_path = Path(path)
    if source_path.suffix == ".csv":
        frame = pd.read_csv(source_path)
        records = frame.to_dict(orient="records")
    elif source_path.suffix == ".jsonl":
        records = []
        with source_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    records.append(json.loads(line))
    else:
        raise ValueError(f"Unsupported dataset format: {source_path.suffix}")

    _validate_records(records)
    return records


def tokenize_records(
    records: Iterable[dict[str, Any]], tokenizer: Any, max_length: int
) -> list[dict[str, Any]]:
    materialized = list(records)
    texts = [str(record["text"]) for record in materialized]
    tokenized = tokenizer(
        texts, padding="max_length", truncation=True, max_length=max_length
    )

    output = []
    for index, record in enumerate(materialized):
        output.append(
            {
                "input_ids": tokenized["input_ids"][index],
                "attention_mask": tokenized["attention_mask"][index],
                "labels": record["label"],
                "difficulty": record["difficulty"],
                "id": record.get("id"),
            }
        )
    return output


@dataclass
class TextDifficultyDataset:
    records: list[dict[str, Any]]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


class UQBatchCollator:
    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        labels = [feature.get("labels") for feature in features]
        difficulty = [feature.get("difficulty") for feature in features]

        label_mask = [label is not None and pd.notna(label) for label in labels]
        difficulty_mask = [
            value is not None and pd.notna(value) for value in difficulty
        ]

        return {
            "input_ids": torch.tensor(
                [feature["input_ids"] for feature in features], dtype=torch.long
            ),
            "attention_mask": torch.tensor(
                [feature["attention_mask"] for feature in features], dtype=torch.long
            ),
            "labels": torch.tensor(
                [float(label) if mask else 0.0 for label, mask in zip(labels, label_mask)],
                dtype=torch.float32,
            ),
            "difficulty": torch.tensor(
                [
                    float(value) if mask else 0.0
                    for value, mask in zip(difficulty, difficulty_mask)
                ],
                dtype=torch.float32,
            ),
            "label_mask": torch.tensor(label_mask, dtype=torch.bool),
            "difficulty_mask": torch.tensor(difficulty_mask, dtype=torch.bool),
        }


def _validate_records(records: list[dict[str, Any]]) -> None:
    for record in records:
        missing = REQUIRED_COLUMNS.difference(record)
        if missing:
            raise ValueError(f"Record missing required columns: {sorted(missing)}")
