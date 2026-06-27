from __future__ import annotations

import io
import textwrap

import pandas as pd
import pytest


def make_pair_csv(tmp_path):
    content = textwrap.dedent("""\
        text_a,text_b,score_a,score_b,margin
        "hello world","goodbye world",0.8,0.2,0.1
        "foo bar","baz qux",0.9,0.3,0.2
    """)
    path = tmp_path / "train_pairs.csv"
    path.write_text(content)
    return path


def test_load_pair_records(tmp_path):
    from modernbert_uq.data import load_pair_records
    path = make_pair_csv(tmp_path)
    records = load_pair_records(path)
    assert len(records) == 2
    assert set(records[0].keys()) >= {"text_a", "text_b", "score_a", "score_b", "margin"}
    assert records[0]["text_a"] == "hello world"
    assert float(records[0]["score_a"]) == pytest.approx(0.8)
    assert float(records[0]["margin"]) == pytest.approx(0.1)


def test_load_pair_records_missing_column(tmp_path):
    from modernbert_uq.data import load_pair_records
    bad = tmp_path / "bad.csv"
    bad.write_text("text_a,text_b\nhello,world\n")
    with pytest.raises(ValueError, match="missing columns"):
        load_pair_records(bad)


def test_pair_dataset_len_and_getitem(tmp_path):
    from modernbert_uq.data import PairDataset
    records = [{"a": 1}, {"a": 2}]
    ds = PairDataset(records)
    assert len(ds) == 2
    assert ds[0] == {"a": 1}


def test_pair_batch_collator():
    pytest.importorskip("torch")
    import torch
    from modernbert_uq.data import PairBatchCollator

    features = [
        {
            "input_ids_a": [1, 2, 3],
            "attention_mask_a": [1, 1, 1],
            "input_ids_b": [4, 5, 6],
            "attention_mask_b": [1, 1, 0],
            "score_a": 0.8,
            "score_b": 0.2,
            "margin": 0.1,
        },
        {
            "input_ids_a": [7, 8, 9],
            "attention_mask_a": [1, 1, 1],
            "input_ids_b": [10, 11, 12],
            "attention_mask_b": [1, 0, 0],
            "score_a": 0.9,
            "score_b": 0.1,
            "margin": 0.2,
        },
    ]
    collator = PairBatchCollator()
    batch = collator(features)
    assert batch["input_ids_a"].shape == (2, 3)
    assert batch["input_ids_b"].shape == (2, 3)
    assert batch["score_a"].dtype == torch.float32
    assert batch["score_b"].shape == (2,)
    assert batch["margin"].shape == (2,)


def test_pair_csv_iterable_dataset_and_text_collator(tmp_path):
    pytest.importorskip("torch")
    from modernbert_uq.data import PairCsvIterableDataset, PairTextBatchCollator

    path = make_pair_csv(tmp_path)
    dataset = PairCsvIterableDataset(path)
    rows = list(dataset)
    assert len(rows) == 2
    assert rows[0]["text_a"] == "hello world"

    class DummyTokenizer:
        def __call__(self, texts, padding, truncation, max_length):
            assert padding == "max_length"
            assert truncation is True
            encoded = []
            masks = []
            for text in texts:
                ids = [len(text) % 10 + 1]
                ids = ids + [0] * (max_length - len(ids))
                mask = [1] + [0] * (max_length - 1)
                encoded.append(ids)
                masks.append(mask)
            return {"input_ids": encoded, "attention_mask": masks}

    collator = PairTextBatchCollator(tokenizer=DummyTokenizer(), max_length=4)
    batch = collator(rows)
    assert batch["input_ids_a"].shape == (2, 4)
    assert batch["input_ids_b"].shape == (2, 4)
    assert batch["margin"].shape == (2,)
