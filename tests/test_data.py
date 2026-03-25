import json
from pathlib import Path

import pytest

from modernbert_uq.data import (
    TextDifficultyDataset,
    UQBatchCollator,
    load_records,
    tokenize_records,
)


class FakeTokenizer:
    def __call__(self, texts, padding, truncation, max_length):
        input_ids = []
        attention_mask = []
        for text in texts:
            token_count = min(len(text.split()), max_length)
            ids = list(range(1, token_count + 1))
            pad_count = max_length - token_count
            input_ids.append(ids + ([0] * pad_count))
            attention_mask.append(([1] * token_count) + ([0] * pad_count))
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def test_data_loading(tmp_path: Path):
    csv_path = tmp_path / "records.csv"
    csv_path.write_text(
        "text,label,difficulty,id\nalpha,1,0.2,a\nbeta,0,0.8,b\n", encoding="utf-8"
    )

    jsonl_path = tmp_path / "records.jsonl"
    jsonl_rows = [
        {"text": "gamma", "label": 1, "difficulty": 0.1, "id": "c"},
        {"text": "delta", "label": None, "difficulty": 0.9, "id": "d"},
    ]
    jsonl_path.write_text(
        "\n".join(json.dumps(row) for row in jsonl_rows), encoding="utf-8"
    )

    csv_records = load_records(csv_path)
    jsonl_records = load_records(jsonl_path)

    assert len(csv_records) == 2
    assert csv_records[0]["text"] == "alpha"
    assert csv_records[1]["label"] == 0
    assert jsonl_records[1]["difficulty"] == pytest.approx(0.9)


@pytest.mark.requires_torch
def test_tokenization_and_collation():
    records = [
        {"text": "short example", "label": 1, "difficulty": 0.3},
        {"text": "a much longer example than before", "label": None, "difficulty": 0.7},
    ]

    tokenized = tokenize_records(records, tokenizer=FakeTokenizer(), max_length=4)
    dataset = TextDifficultyDataset(tokenized)
    batch = UQBatchCollator()(list(dataset))

    assert tokenized[0]["input_ids"] == [1, 2, 0, 0]
    assert tokenized[1]["attention_mask"] == [1, 1, 1, 1]
    assert batch["input_ids"].shape == (2, 4)
    assert batch["attention_mask"].shape == (2, 4)
    assert batch["labels"].shape == (2,)
    assert batch["difficulty"].shape == (2,)
    assert batch["label_mask"].tolist() == [True, False]
