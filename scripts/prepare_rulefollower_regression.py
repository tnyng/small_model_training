from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from prepare_rulefollower_branch import (
    BRANCHES,
    assign_split,
    build_pair_frame,
    build_text_fields,
)

BIN_LABELS = ("1-4", "5-7", "8-9")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare rebalanced RuleFollower regression datasets."
    )
    parser.add_argument(
        "--branch",
        required=True,
        choices=sorted(BRANCHES),
        help="Branch key to prepare.",
    )
    parser.add_argument(
        "--variant",
        required=True,
        choices=("reg_base", "reg_down_exact", "reg_over_exact", "reg_down_bins", "reg_over_bins"),
        help="Regression rebalancing variant.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="RuleFollower project root.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
        help="Directory where prepared datasets will be written.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Optional dataset directory name under output-root.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Deterministic sampling seed.")
    return parser.parse_args()


def assign_score_target(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["score_target"] = (result["accuracy_score"].astype(float) - 1.0) / 8.0
    result["score_bucket"] = result["accuracy_score"].astype(int)
    result["score_bin"] = result["score_bucket"].map(score_to_bin)
    return result


def score_to_bin(score: int) -> str:
    score = int(score)
    if 1 <= score <= 4:
        return "1-4"
    if 5 <= score <= 7:
        return "5-7"
    if 8 <= score <= 9:
        return "8-9"
    raise ValueError(f"Unexpected score: {score}")


def rebalance_train_split(frame: pd.DataFrame, variant: str, seed: int) -> pd.DataFrame:
    train = frame.loc[frame["split"] == "train"].copy()
    validation_test = frame.loc[frame["split"] != "train"].copy()

    if variant == "reg_base":
        return frame
    if variant == "reg_down_exact":
        train = rebalance_by_group(train, group_col="score_bucket", mode="down", seed=seed)
    elif variant == "reg_over_exact":
        train = rebalance_by_group(train, group_col="score_bucket", mode="over", seed=seed)
    elif variant == "reg_down_bins":
        train = rebalance_by_group(train, group_col="score_bin", mode="down", seed=seed)
    elif variant == "reg_over_bins":
        train = rebalance_by_group(train, group_col="score_bin", mode="over", seed=seed)
    else:
        raise ValueError(f"Unsupported variant: {variant}")

    combined = pd.concat([train, validation_test], ignore_index=True)
    return combined


def rebalance_by_group(
    frame: pd.DataFrame,
    group_col: str,
    mode: str,
    seed: int,
) -> pd.DataFrame:
    grouped = {group: sub.copy() for group, sub in frame.groupby(group_col, sort=True)}
    counts = {group: len(sub) for group, sub in grouped.items() if len(sub) > 0}
    if not counts:
        return frame

    target_size = min(counts.values()) if mode == "down" else max(counts.values())
    sampled = []
    for offset, (group, sub) in enumerate(sorted(grouped.items(), key=lambda item: item[0])):
        replace = mode == "over" and len(sub) < target_size
        sampled.append(
            sub.sample(
                n=target_size,
                replace=replace,
                random_state=seed + offset,
            )
        )
    return pd.concat(sampled, ignore_index=True)


def write_split_files(
    frame: pd.DataFrame,
    output_dir: Path,
    variant: str,
    original_train_frame: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = [
        "id",
        "text",
        "score_target",
        "difficulty",
        "split",
        "dataset",
        "parser_type",
        "parser_model",
        "annotator_model",
        "accuracy_score",
        "difficulty_raw",
        "score_bucket",
        "score_bin",
        "text_id",
        "source_text",
        "property",
    ]

    for split_name in ("train", "validation", "test"):
        split_frame = frame.loc[frame["split"] == split_name, columns]
        split_frame.to_csv(output_dir / f"{split_name}.csv", index=False)

    train_frame = frame.loc[frame["split"] == "train"].copy()
    summary = {
        "rows": int(len(frame)),
        "unique_texts": int(frame["text_id"].nunique()),
        "variant": variant,
        "rebalancing_mode": variant,
        "score_target_formula": "(accuracy_score - 1) / 8",
        "bin_definition": {"1-4": [1, 4], "5-7": [5, 7], "8-9": [8, 9]},
        "score_distribution_all": {
            str(key): int(value)
            for key, value in frame["score_bucket"].value_counts().sort_index().items()
        },
        "score_distribution_train_original": {
            str(key): int(value)
            for key, value in original_train_frame["score_bucket"].value_counts().sort_index().items()
        },
        "score_distribution_train": {
            str(key): int(value)
            for key, value in train_frame["score_bucket"].value_counts().sort_index().items()
        },
        "bin_distribution_train_original": {
            key: int(value)
            for key, value in original_train_frame["score_bin"].value_counts().sort_index().items()
        },
        "bin_distribution_train": {
            key: int(value)
            for key, value in train_frame["score_bin"].value_counts().sort_index().items()
        },
        "splits": {
            key: int(value)
            for key, value in frame["split"].value_counts().sort_index().items()
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_name = args.output_name or f"{args.branch}_{args.variant}"

    frame = build_pair_frame(root=root, branch_key=args.branch)
    frame = assign_score_target(frame)
    frame = build_text_fields(frame)
    frame = assign_split(frame, seed=args.seed)
    original_train_frame = frame.loc[frame["split"] == "train"].copy()
    frame = rebalance_train_split(frame, variant=args.variant, seed=args.seed)

    output_dir = args.output_root / output_name
    write_split_files(
        frame,
        output_dir,
        variant=args.variant,
        original_train_frame=original_train_frame,
    )
    print(f"Wrote prepared regression dataset to {output_dir}")


if __name__ == "__main__":
    main()
