from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd


DIFFICULTY_RE = re.compile(
    r"<final_difficulty_score>\s*([0-9]*\.?[0-9]+)\s*</final_difficulty_score>",
    re.IGNORECASE,
)
SUPPORTEDNESS_RE = re.compile(r"\[Answer\]:\s*([1-9])\b")

DATASETS = [
    "article_bias",
    "bureaucracies",
    "gwsd",
    "hatecot",
    "implicit_hate",
    "misinfo",
    "misinfo_cancer",
    "polite_stack",
    "polite_wiki",
    "rumoureval2019",
    "tweets",
    "tweets23",
    "tweets_news",
    "tweetscongress",
    "yelp",
]

BRANCHES = {
    "llm_gptoss__ann_gptoss": {
        "parser_type": "llm",
        "parser_model": "gpt-oss",
        "annotator_model": "gpt-oss",
        "root": "experiments/annotation_gpt-oss",
        "accuracy_dir": "accuracy_outputs",
        "accuracy_suffix": "_score_gpt_oss_annotated.csv",
        "difficulty_dir": "difficulty_outputs",
        "difficulty_suffix": "_difficulty_gpt_oss.csv",
    },
    "llm_gptoss__ann_gpt41mini": {
        "parser_type": "llm",
        "parser_model": "gpt-oss",
        "annotator_model": "gpt-4.1-mini",
        "root": "experiments/annotation_gpt4.1mini_gpt-oss",
        "accuracy_dir": "accuracy_outputs",
        "accuracy_suffix": "_score_gpt-4.1-mini.csv",
        "difficulty_dir": "difficulty_outputs",
        "difficulty_suffix": "_difficulty_gpt-4.1-mini.csv",
    },
    "llm_gpt5mini__ann_gpt5mini": {
        "parser_type": "llm",
        "parser_model": "gpt-5-mini",
        "annotator_model": "gpt-5-mini",
        "root": "experiments/annotation_gpt5mini",
        "accuracy_dir": "accuracy_outputs",
        "accuracy_suffix": "_score_gpt-5-mini.csv",
        "difficulty_dir": "difficulty_outputs",
        "difficulty_suffix": "_difficulty_gpt-5-mini.csv",
    },
    "llm_gpt5mini__ann_gpt41mini": {
        "parser_type": "llm",
        "parser_model": "gpt-5-mini",
        "annotator_model": "gpt-4.1-mini",
        "root": "experiments/annotation_gpt4.1mini_gpt-5-mini",
        "accuracy_dir": "accuracy_outputs",
        "accuracy_suffix": "_score_gpt-4.1-mini.csv",
        "difficulty_dir": "difficulty_outputs",
        "difficulty_suffix": "_difficulty_gpt-4.1-mini.csv",
    },
    "heuristic__ann_gptoss": {
        "parser_type": "heuristic",
        "parser_model": "heuristic",
        "annotator_model": "gpt-oss",
        "root": "experiments/annotation_gpt-oss_heuristic",
        "accuracy_dir": "accuracy_outputs",
        "accuracy_suffix": "_score_gpt_oss_annotated.csv",
        "difficulty_dir": "difficulty_outputs",
        "difficulty_suffix": "_difficulty_gpt_oss.csv",
    },
    "heuristic__ann_gpt41mini": {
        "parser_type": "heuristic",
        "parser_model": "heuristic",
        "annotator_model": "gpt-4.1-mini",
        "root": "experiments/annotation_gpt4.1mini_heuristic",
        "accuracy_dir": "accuracy_outputs",
        "accuracy_suffix": "_score_gpt-4.1-mini.csv",
        "difficulty_dir": "difficulty_outputs",
        "difficulty_suffix": "_difficulty_gpt-4.1-mini.csv",
    },
    "heuristic__ann_gpt5mini": {
        "parser_type": "heuristic",
        "parser_model": "heuristic",
        "annotator_model": "gpt-5-mini",
        "root": "experiments/annotation_gpt5mini_heuristic",
        "accuracy_dir": "accuracy_outputs",
        "accuracy_suffix": "_score_gpt-5-mini.csv",
        "difficulty_dir": "difficulty_outputs",
        "difficulty_suffix": "_difficulty_gpt-5-mini.csv",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a RuleFollower branch dataset for ModernBERT training."
    )
    parser.add_argument(
        "--branch",
        required=True,
        choices=sorted(BRANCHES),
        help="Branch key to prepare.",
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
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the deterministic split.",
    )
    parser.add_argument(
        "--accuracy-threshold",
        type=float,
        default=8.0,
        help="Positive label threshold.",
    )
    parser.add_argument(
        "--label-rule",
        choices=("gt", "ge"),
        default="gt",
        help="Binary label rule: `gt` means accuracy_score > threshold; `ge` means accuracy_score >= threshold.",
    )
    return parser.parse_args()


def extract_final_difficulty(raw_output: object) -> float | None:
    if not isinstance(raw_output, str):
        return None
    match = DIFFICULTY_RE.search(raw_output)
    if not match:
        return None
    return float(match.group(1))


def normalize_final_difficulty(score: float) -> float:
    return (float(score) - 1.0) / 4.0


def extract_supportedness(row: pd.Series) -> float | None:
    if "weighted_scale_sums" in row and pd.notna(row["weighted_scale_sums"]):
        try:
            return float(row["weighted_scale_sums"])
        except Exception:
            pass
    raw_output = row.get("output")
    if not isinstance(raw_output, str):
        return None
    match = SUPPORTEDNESS_RE.search(raw_output)
    if not match:
        return None
    return float(match.group(1))


def build_pair_frame(root: Path, branch_key: str) -> pd.DataFrame:
    branch = BRANCHES[branch_key]
    frames: list[pd.DataFrame] = []
    branch_root = root / branch["root"]

    for dataset in DATASETS:
        accuracy_path = (
            branch_root
            / branch["accuracy_dir"]
            / f"{dataset}{branch['accuracy_suffix']}"
        )
        difficulty_path = (
            branch_root
            / branch["difficulty_dir"]
            / f"{dataset}{branch['difficulty_suffix']}"
        )
        if not accuracy_path.exists() or not difficulty_path.exists():
            continue

        accuracy_frame = pd.read_csv(accuracy_path).copy()
        accuracy_frame["accuracy_score"] = accuracy_frame.apply(extract_supportedness, axis=1)
        accuracy_frame = accuracy_frame.dropna(subset=["accuracy_score"])

        difficulty_frame = pd.read_csv(difficulty_path).copy()
        difficulty_frame["difficulty_raw"] = difficulty_frame["output"].map(
            extract_final_difficulty
        )
        difficulty_frame = difficulty_frame.dropna(subset=["difficulty_raw"])
        difficulty_frame["difficulty"] = difficulty_frame["difficulty_raw"].map(
            normalize_final_difficulty
        )

        merged = accuracy_frame.merge(
            difficulty_frame[["text", "property", "file_name", "id", "difficulty", "difficulty_raw"]],
            on=["text", "property", "file_name", "id"],
            how="inner",
            validate="one_to_one",
        )
        merged["dataset"] = dataset
        frames.append(merged)

    if not frames:
        raise ValueError(f"No matched accuracy/difficulty files found for branch {branch_key}")

    pair_frame = pd.concat(frames, ignore_index=True)
    pair_frame["parser_type"] = branch["parser_type"]
    pair_frame["parser_model"] = branch["parser_model"]
    pair_frame["annotator_model"] = branch["annotator_model"]
    return pair_frame


def assign_label(frame: pd.DataFrame, threshold: float, label_rule: str) -> pd.DataFrame:
    result = frame.copy()
    if label_rule == "gt":
        result["label"] = (result["accuracy_score"] > float(threshold)).astype(int)
    elif label_rule == "ge":
        result["label"] = (result["accuracy_score"] >= float(threshold)).astype(int)
    else:
        raise ValueError(f"Unsupported label_rule: {label_rule}")
    return result


def build_text_fields(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["source_text"] = result["text"]
    result["text"] = result.apply(
        lambda row: f"Text: {row['source_text']}\n\nClaim: {row['property']}",
        axis=1,
    )
    result["text_id"] = result.apply(
        lambda row: f"{row['dataset']}__"
        f"{hashlib.sha1(str(row['source_text']).encode('utf-8')).hexdigest()[:16]}",
        axis=1,
    )
    result["id"] = result.apply(
        lambda row: f"{row['text_id']}__claim_{int(row.name)}",
        axis=1,
    )
    return result


def assign_split(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    text_units = (
        frame[["text_id"]]
        .drop_duplicates(subset=["text_id"], keep="first")
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )
    total = len(text_units)
    if total < 3:
        raise ValueError(f"Need at least 3 unique texts after merging, found {total}")

    train_count = max(1, int(round(total * 0.70)))
    validation_count = max(1, int(round(total * 0.15)))
    test_count = total - train_count - validation_count
    if test_count < 1:
        deficit = 1 - test_count
        if train_count >= validation_count and train_count - deficit >= 1:
            train_count -= deficit
        else:
            validation_count -= deficit
        test_count = 1

    split_labels = (
        (["train"] * train_count)
        + (["validation"] * validation_count)
        + (["test"] * test_count)
    )
    if len(split_labels) != len(text_units):
        raise ValueError(
            f"Split allocation mismatch: rows={len(text_units)} labels={len(split_labels)}"
        )

    text_units["split"] = split_labels
    return frame.merge(
        text_units[["text_id", "split"]],
        on="text_id",
        how="left",
        validate="many_to_one",
    )


def write_split_files(frame: pd.DataFrame, output_dir: Path, threshold: float, label_rule: str) -> None:
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

    summary = {
        "rows": int(len(frame)),
        "unique_texts": int(frame["text_id"].nunique()),
        "label_rule": f"accuracy_score {'>' if label_rule == 'gt' else '>='} {threshold}",
        "label_distribution": {
            str(key): int(value)
            for key, value in frame["label"].value_counts().sort_index().items()
        },
        "accuracy_min": float(frame["accuracy_score"].min()),
        "accuracy_max": float(frame["accuracy_score"].max()),
        "difficulty_min": float(frame["difficulty"].min()),
        "difficulty_max": float(frame["difficulty"].max()),
        "datasets": {
            key: int(value)
            for key, value in frame["dataset"].value_counts().sort_index().items()
        },
        "splits": {
            key: int(value)
            for key, value in frame["split"].value_counts().sort_index().items()
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_name = args.output_name or args.branch

    frame = build_pair_frame(root=root, branch_key=args.branch)
    frame = assign_label(
        frame,
        threshold=args.accuracy_threshold,
        label_rule=args.label_rule,
    )
    frame = build_text_fields(frame)
    frame = assign_split(frame, seed=args.seed)

    output_dir = args.output_root / output_name
    write_split_files(
        frame,
        output_dir,
        threshold=args.accuracy_threshold,
        label_rule=args.label_rule,
    )
    print(f"Wrote prepared dataset to {output_dir}")


if __name__ == "__main__":
    main()
