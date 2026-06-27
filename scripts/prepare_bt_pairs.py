#!/usr/bin/env python3
"""Generate a Bradley-Terry pair CSV from a regression score CSV.

Reads train.csv from --input-dir, generates unique pairs (i, j) where
|score_i - score_j| >= --threshold (with score_i > score_j), and writes
train_pairs.csv to --output-dir. When --max-pairs is positive, the script
samples reproducibly without materializing the full O(n^2) candidate set.
Copies validation/test splits unchanged.

Usage:
    python scripts/prepare_bt_pairs.py \
        --input-dir data/rulefollower_heuristic_ann_gpt41mini_reg_base \
        --output-dir data/rulefollower_heuristic_ann_gpt41mini_bt_base \
        --threshold 0.1
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate BT pair CSV from regression CSV.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="Minimum |score_a - score_b| to include a pair (default: 0.1)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--score-column",
        type=str,
        default="score_target",
        help="Column name for regression score (default: score_target)",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=2000000,
        help="Maximum number of pairs to generate (default: 2000000). If 0, generate all valid pairs.",
    )
    parser.add_argument(
        "--margin-mode",
        type=str,
        default="none",
        choices=["none", "llama2_gap"],
        help="Optional margin column to add. 'llama2_gap' maps larger raw score gaps to larger discrete margins.",
    )
    return parser.parse_args()


def generate_pairs(
    frame: pd.DataFrame,
    score_column: str,
    threshold: float,
    rng: np.random.Generator,
    max_pairs: int | None = None,
) -> pd.DataFrame:
    texts = frame["text"].to_numpy()
    scores = frame[score_column].to_numpy(dtype=float)
    n = len(texts)

    if max_pairs is None or max_pairs <= 0:
        return _generate_all_pairs(texts=texts, scores=scores, threshold=threshold, rng=rng)

    return _sample_pairs(
        texts=texts,
        scores=scores,
        threshold=threshold,
        rng=rng,
        max_pairs=max_pairs,
    )


def _generate_all_pairs(
    texts: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    preferred_indices: list[int] = []
    rejected_indices: list[int] = []
    n = len(texts)
    for i in range(n):
        for j in range(i + 1, n):
            diff = scores[i] - scores[j]
            if abs(diff) < threshold:
                continue
            if diff > 0:
                preferred_indices.append(i)
                rejected_indices.append(j)
            else:
                preferred_indices.append(j)
                rejected_indices.append(i)

    if not preferred_indices:
        raise ValueError(
            f"No pairs found with |score_a - score_b| >= {threshold}. "
            "Try a lower threshold."
        )

    order = rng.permutation(len(preferred_indices))
    preferred = np.asarray(preferred_indices, dtype=np.int64)[order]
    rejected = np.asarray(rejected_indices, dtype=np.int64)[order]
    return _build_pair_frame(texts=texts, scores=scores, preferred=preferred, rejected=rejected)


def _sample_pairs(
    texts: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    rng: np.random.Generator,
    max_pairs: int,
) -> pd.DataFrame:
    n = len(texts)
    seen_pairs: set[tuple[int, int]] = set()
    preferred_indices: list[int] = []
    rejected_indices: list[int] = []

    # Rejection-sample unordered pairs, then keep only those above threshold.
    # This avoids constructing the full O(n^2) candidate set on large datasets.
    while len(preferred_indices) < max_pairs:
        remaining = max_pairs - len(preferred_indices)
        batch_size = min(max(remaining * 8, 50_000), 1_000_000)

        left = rng.integers(0, n, size=batch_size)
        right = rng.integers(0, n - 1, size=batch_size)
        right = right + (right >= left)

        low = np.minimum(left, right)
        high = np.maximum(left, right)
        diff = scores[low] - scores[high]
        valid = np.abs(diff) >= threshold

        if not np.any(valid):
            continue

        valid_low = low[valid]
        valid_high = high[valid]
        valid_diff = diff[valid]

        for low_idx, high_idx, pair_diff in zip(
            valid_low.tolist(),
            valid_high.tolist(),
            valid_diff.tolist(),
        ):
            pair_key = (low_idx, high_idx)
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            if pair_diff > 0:
                preferred_indices.append(low_idx)
                rejected_indices.append(high_idx)
            else:
                preferred_indices.append(high_idx)
                rejected_indices.append(low_idx)
            if len(preferred_indices) >= max_pairs:
                break

    preferred = np.asarray(preferred_indices, dtype=np.int64)
    rejected = np.asarray(rejected_indices, dtype=np.int64)
    return _build_pair_frame(texts=texts, scores=scores, preferred=preferred, rejected=rejected)


def _build_pair_frame(
    texts: np.ndarray,
    scores: np.ndarray,
    preferred: np.ndarray,
    rejected: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "text_a": texts[preferred],
            "text_b": texts[rejected],
            "score_a": scores[preferred],
            "score_b": scores[rejected],
        }
    )


def _compute_margin(score_a: pd.Series, score_b: pd.Series, margin_mode: str) -> pd.Series:
    if margin_mode == "none":
        return pd.Series([0.0] * len(score_a), index=score_a.index, dtype=float)
    if margin_mode != "llama2_gap":
        raise ValueError(f"Unsupported margin mode: {margin_mode}")

    raw_gap = (score_a - score_b).abs()
    # Llama 2 uses a discrete margin based on preference strength.
    # Here we use a conservative piecewise mapping on the raw 1-9 score gap
    # to avoid demanding impossible score separations from a sigmoid-bounded model.
    margins = pd.Series(0.0, index=score_a.index, dtype=float)
    margins = margins.mask(raw_gap >= 2, 0.10)
    margins = margins.mask(raw_gap >= 3, 0.20)
    margins = margins.mask(raw_gap >= 4, 0.30)
    return margins


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_path = args.input_dir / "train.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"train.csv not found in {args.input_dir}")

    train_frame = pd.read_csv(train_path)
    print(f"[INFO] Loaded {len(train_frame)} training rows from {train_path}")

    pairs = generate_pairs(train_frame, args.score_column, args.threshold, rng, args.max_pairs)
    if args.margin_mode != "none":
        pairs["margin"] = _compute_margin(
            pairs["score_a"],
            pairs["score_b"],
            margin_mode=args.margin_mode,
        )
    out_path = args.output_dir / "train_pairs.csv"
    pairs.to_csv(out_path, index=False)
    print(f"[INFO] Wrote {len(pairs)} pairs to {out_path}")

    score_diffs = (pairs["score_a"] - pairs["score_b"]).describe()
    print(f"[INFO] Score-diff distribution:\n{score_diffs}")

    for split in ("val.csv", "test.csv", "validation.csv"):
        src = args.input_dir / split
        if src.exists():
            dst = args.output_dir / split
            shutil.copy2(src, dst)
            print(f"[INFO] Copied {src.name} -> {dst}")

    import json
    summary = {
        "input_dir": str(args.input_dir),
        "threshold": args.threshold,
        "train_rows_in": int(len(train_frame)),
        "pairs_generated": int(len(pairs)),
        "seed": args.seed,
        "max_pairs": args.max_pairs,
    }
    (args.output_dir / "pair_summary.json").write_text(json.dumps(summary, indent=2))
    print("[DONE] Pair dataset ready.")


if __name__ == "__main__":
    main()
