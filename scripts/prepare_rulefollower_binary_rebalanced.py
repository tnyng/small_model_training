from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from prepare_rulefollower_branch import (
    BRANCHES,
    assign_label,
    assign_split,
    build_pair_frame,
    build_text_fields,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare GE7 binary datasets with label-level rebalancing."
    )
    parser.add_argument("--branch", required=True, choices=sorted(BRANCHES))
    parser.add_argument(
        "--variant",
        required=True,
        choices=("cls_base", "cls_down_label", "cls_over_label"),
        help="Binary rebalancing variant.",
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
    parser.add_argument("--output-name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--accuracy-threshold",
        type=float,
        default=7.0,
        help="GE7 by default: positive if accuracy_score >= 7.",
    )
    parser.add_argument(
        "--label-rule",
        choices=("ge",),
        default="ge",
        help="Binary rule is fixed to >= threshold for this experiment.",
    )
    return parser.parse_args()


def rebalance_train_split(frame: pd.DataFrame, variant: str, seed: int) -> pd.DataFrame:
    train = frame.loc[frame["split"] == "train"].copy()
    holdout = frame.loc[frame["split"] != "train"].copy()

    if variant == "cls_base":
        return frame

    label_counts = train["label"].value_counts().sort_index()
    if len(label_counts) < 2:
        raise ValueError(
            f"Need both classes in the training split for rebalancing, got {label_counts.to_dict()}"
        )

    target_size = int(label_counts.min()) if variant == "cls_down_label" else int(label_counts.max())
    samples = []
    for offset, label in enumerate(sorted(label_counts.index.tolist())):
        sub = train.loc[train["label"] == label].copy()
        replace = variant == "cls_over_label" and len(sub) < target_size
        samples.append(
            sub.sample(
                n=target_size,
                replace=replace,
                random_state=seed + offset,
            )
        )

    rebalanced_train = pd.concat(samples, ignore_index=True)
    return pd.concat([rebalanced_train, holdout], ignore_index=True)


def compute_class_weights(train_frame: pd.DataFrame) -> dict[str, float]:
    counts = train_frame["label"].value_counts().sort_index().to_dict()
    negatives = float(counts.get(0, 0))
    positives = float(counts.get(1, 0))
    total = negatives + positives
    if negatives == 0 or positives == 0:
        return {
            "positive_class_weight": 1.0,
            "negative_class_weight": 1.0,
        }
    return {
        "positive_class_weight": total / (2.0 * positives),
        "negative_class_weight": total / (2.0 * negatives),
    }


def write_split_files(
    frame: pd.DataFrame,
    output_dir: Path,
    threshold: float,
    label_rule: str,
    variant: str,
    original_train_frame: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = [
        "id",
        "text",
        "label",
        "difficulty",
        "split",
        "dataset",
        "parser_type",
        "parser_model",
        "annotator_model",
        "accuracy_score",
        "difficulty_raw",
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
        "label_rule": f"accuracy_score {'>=' if label_rule == 'ge' else '>'} {threshold}",
        "label_distribution_all": {
            str(key): int(value)
            for key, value in frame["label"].value_counts().sort_index().items()
        },
        "label_distribution_train_original": {
            str(key): int(value)
            for key, value in original_train_frame["label"].value_counts().sort_index().items()
        },
        "label_distribution_train": {
            str(key): int(value)
            for key, value in train_frame["label"].value_counts().sort_index().items()
        },
        "class_weights_from_original_train": compute_class_weights(original_train_frame),
        "class_weights_from_final_train": compute_class_weights(train_frame),
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
    frame = assign_label(frame, threshold=args.accuracy_threshold, label_rule=args.label_rule)
    frame = build_text_fields(frame)
    frame = assign_split(frame, seed=args.seed)
    original_train_frame = frame.loc[frame["split"] == "train"].copy()
    frame = rebalance_train_split(frame, variant=args.variant, seed=args.seed)

    write_split_files(
        frame,
        args.output_root / output_name,
        threshold=args.accuracy_threshold,
        label_rule=args.label_rule,
        variant=args.variant,
        original_train_frame=original_train_frame,
    )
    print(f"Wrote prepared binary rebalanced dataset to {args.output_root / output_name}")


if __name__ == "__main__":
    main()
