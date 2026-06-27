from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))

from prepare_bt_pairs import generate_pairs


def test_generate_pairs_respects_threshold_and_order():
    frame = pd.DataFrame(
        {
            "text": ["a", "b", "c"],
            "score_target": [0.9, 0.2, 0.82],
        }
    )
    pairs = generate_pairs(
        frame=frame,
        score_column="score_target",
        threshold=0.1,
        rng=np.random.default_rng(42),
        max_pairs=0,
    )
    assert not pairs.empty
    assert (pairs["score_a"] > pairs["score_b"]).all()
    assert ((pairs["score_a"] - pairs["score_b"]) >= 0.1).all()


def test_generate_pairs_sampling_is_reproducible():
    frame = pd.DataFrame(
        {
            "text": [f"text-{idx}" for idx in range(200)],
            "score_target": np.linspace(0.0, 1.0, 200),
        }
    )
    pairs_a = generate_pairs(
        frame=frame,
        score_column="score_target",
        threshold=0.1,
        rng=np.random.default_rng(7),
        max_pairs=250,
    )
    pairs_b = generate_pairs(
        frame=frame,
        score_column="score_target",
        threshold=0.1,
        rng=np.random.default_rng(7),
        max_pairs=250,
    )
    assert pairs_a.equals(pairs_b)
