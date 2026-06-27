from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

try:
    from torch.utils.data import IterableDataset
except ModuleNotFoundError:  # pragma: no cover - exercised in non-torch test envs
    class IterableDataset:  # type: ignore[no-redef]
        pass

BASE_REQUIRED_COLUMNS = {"text", "difficulty"}


def load_records(path: str | Path, target_column: str = "label") -> list[dict[str, Any]]:
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

    _validate_records(records, target_column=target_column)
    return records


def tokenize_records(
    records: Iterable[dict[str, Any]],
    tokenizer: Any,
    max_length: int,
    target_column: str = "label",
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
                "labels": record[target_column],
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


def _validate_records(records: list[dict[str, Any]], target_column: str) -> None:
    for record in records:
        missing = (BASE_REQUIRED_COLUMNS | {target_column}).difference(record)
        if missing:
            raise ValueError(f"Record missing required columns: {sorted(missing)}")


def load_pair_records(path: str | Path) -> list[dict[str, Any]]:
    """Load a pair CSV with columns text_a, text_b, score_a, score_b."""
    source_path = Path(path)
    frame = pd.read_csv(source_path)
    required = {"text_a", "text_b", "score_a", "score_b"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Pair CSV missing columns: {sorted(missing)}")
    return frame.to_dict(orient="records")


def tokenize_pair_records(
    records: list[dict[str, Any]],
    tokenizer: Any,
    max_length: int,
) -> list[dict[str, Any]]:
    """Tokenize both texts in each pair record."""
    texts_a = [str(r["text_a"]) for r in records]
    texts_b = [str(r["text_b"]) for r in records]
    tok_a = tokenizer(texts_a, padding="max_length", truncation=True, max_length=max_length)
    tok_b = tokenizer(texts_b, padding="max_length", truncation=True, max_length=max_length)
    output = []
    for i, record in enumerate(records):
        output.append(
            {
                "input_ids_a": tok_a["input_ids"][i],
                "attention_mask_a": tok_a["attention_mask"][i],
                "input_ids_b": tok_b["input_ids"][i],
                "attention_mask_b": tok_b["attention_mask"][i],
                "score_a": float(record["score_a"]),
                "score_b": float(record["score_b"]),
                "margin": float(record["margin"]) if "margin" in record else None,
            }
        )
    return output


@dataclass
class PairDataset:
    records: list[dict[str, Any]]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


class PairBatchCollator:
    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        batch = {
            "input_ids_a": torch.tensor(
                [f["input_ids_a"] for f in features], dtype=torch.long
            ),
            "attention_mask_a": torch.tensor(
                [f["attention_mask_a"] for f in features], dtype=torch.long
            ),
            "input_ids_b": torch.tensor(
                [f["input_ids_b"] for f in features], dtype=torch.long
            ),
            "attention_mask_b": torch.tensor(
                [f["attention_mask_b"] for f in features], dtype=torch.long
            ),
            "score_a": torch.tensor([f["score_a"] for f in features], dtype=torch.float32),
            "score_b": torch.tensor([f["score_b"] for f in features], dtype=torch.float32),
        }
        if any(f.get("margin") is not None for f in features):
            batch["margin"] = torch.tensor(
                [float(f.get("margin", 0.0) or 0.0) for f in features],
                dtype=torch.float32,
            )
        return batch


class PairCsvIterableDataset(IterableDataset):
    """Stream BT pairs from CSV to avoid materializing multi-GB pair files in memory."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._validate_header()

    def _validate_header(self) -> None:
        with self.path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = set(reader.fieldnames or [])
        required = {"text_a", "text_b", "score_a", "score_b"}
        missing = required - fieldnames
        if missing:
            raise ValueError(f"Pair CSV missing columns: {sorted(missing)}")

    def __iter__(self):
        with self.path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                record = {
                    "text_a": row["text_a"],
                    "text_b": row["text_b"],
                    "score_a": float(row["score_a"]),
                    "score_b": float(row["score_b"]),
                }
                if "margin" in row and row["margin"] not in (None, ""):
                    record["margin"] = float(row["margin"])
                yield record


class PairTextBatchCollator:
    """Tokenize BT text pairs at batch time to keep memory bounded."""

    def __init__(self, tokenizer: Any, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        texts_a = [str(feature["text_a"]) for feature in features]
        texts_b = [str(feature["text_b"]) for feature in features]
        tok_a = self.tokenizer(
            texts_a,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        tok_b = self.tokenizer(
            texts_b,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        batch = {
            "input_ids_a": torch.tensor(tok_a["input_ids"], dtype=torch.long),
            "attention_mask_a": torch.tensor(tok_a["attention_mask"], dtype=torch.long),
            "input_ids_b": torch.tensor(tok_b["input_ids"], dtype=torch.long),
            "attention_mask_b": torch.tensor(tok_b["attention_mask"], dtype=torch.long),
            "score_a": torch.tensor([f["score_a"] for f in features], dtype=torch.float32),
            "score_b": torch.tensor([f["score_b"] for f in features], dtype=torch.float32),
        }
        if any(f.get("margin") is not None for f in features):
            batch["margin"] = torch.tensor(
                [float(f.get("margin", 0.0) or 0.0) for f in features],
                dtype=torch.float32,
            )
        return batch
