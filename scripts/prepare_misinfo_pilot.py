from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


DIFFICULTY_RE = re.compile(
    r"<final_difficulty_score>\s*([0-9]*\.?[0-9]+)\s*</final_difficulty_score>",
    re.IGNORECASE,
)
SUPPORTEDNESS_RE = re.compile(r"\[Answer\]:\s*([1-9])")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare misinfo pilot datasets for ModernBERT training."
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
        "--seed",
        type=int,
        default=42,
        help="Random seed for the deterministic split.",
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


def extract_supportedness(raw_output: object) -> float | None:
    if not isinstance(raw_output, str):
        return None
    match = SUPPORTEDNESS_RE.search(raw_output)
    if not match:
        return None
    return float(match.group(1))


def build_variant_frame(
    cot_frame: pd.DataFrame,
    score_path: Path,
    difficulty_path: Path,
) -> pd.DataFrame:
    cot_frame = cot_frame.rename(columns={"id": "text_id"}).copy()
    score_frame = pd.read_csv(score_path).copy()
    score_frame["supportedness_score"] = score_frame["output"].map(extract_supportedness)
    score_frame = score_frame.dropna(subset=["supportedness_score"])

    difficulty_frame = pd.read_csv(difficulty_path).copy()
    difficulty_frame["difficulty"] = difficulty_frame["output"].map(extract_final_difficulty)
    difficulty_frame = difficulty_frame.dropna(subset=["difficulty"])
    difficulty_frame["difficulty"] = difficulty_frame["difficulty"].map(normalize_final_difficulty)
    pair_frame = score_frame.merge(
        difficulty_frame[["text", "property", "file_name", "id", "difficulty"]],
        on=["text", "property", "file_name", "id"],
        how="inner",
        validate="one_to_one",
    )
    pair_frame["label"] = (pair_frame["supportedness_score"] >= 5.0).astype(int)

    merged = cot_frame.merge(
        pair_frame[["text", "property", "file_name", "id", "label", "difficulty", "supportedness_score"]],
        on="text",
        how="inner",
        validate="one_to_many",
    )
    merged = merged.rename(columns={"text": "source_text"})
    merged["model_input"] = merged.apply(
        lambda row: f"Text: {row['source_text']}\n\nClaim: {row['property']}",
        axis=1,
    )
    merged["id"] = merged.apply(
        lambda row: f"{row['text_id']}__claim_{int(row.name)}",
        axis=1,
    )
    return merged[
        [
            "id",
            "model_input",
            "label",
            "difficulty",
            "text_id",
            "source_text",
            "property",
            "supportedness_score",
        ]
    ].rename(columns={"model_input": "text"})


def assign_split(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    text_units = (
        frame[["text_id"]]
        .drop_duplicates(subset=["text_id"], keep="first")
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )
    total = len(text_units)
    if total < 3:
        raise ValueError(f"Need at least 3 rows after merging, found {total}")
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
    if len(text_units) != len(split_labels):
        raise ValueError(
            f"Split allocation mismatch: rows={len(text_units)} labels={len(split_labels)}"
        )
    text_units["split"] = split_labels
    return frame.merge(text_units[["text_id", "split"]], on="text_id", how="left", validate="many_to_one")


def write_split_files(frame: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name in ("train", "validation", "test"):
        split_frame = frame.loc[
            frame["split"] == split_name, ["id", "text", "label", "difficulty", "split"]
        ]
        split_frame.to_csv(output_dir / f"{split_name}.csv", index=False)
    summary = {
        "rows": int(len(frame)),
        "unique_texts": int(frame["text_id"].nunique()),
        "label_distribution": {
            str(key): int(value) for key, value in frame["label"].value_counts().sort_index().items()
        },
        "difficulty_min": float(frame["difficulty"].min()),
        "difficulty_max": float(frame["difficulty"].max()),
        "supportedness_min": float(frame["supportedness_score"].min()),
        "supportedness_max": float(frame["supportedness_score"].max()),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = args.root.resolve()

    cot_path = root / "experiments/cot_generation/qwen3_32B_misinfo_cot.csv"
    llm_score_path = (
        root / "experiments/step3_gpt4.1mini_api/score_outputs/misinfo_score_gpt-4.1-mini.csv"
    )
    heuristic_score_path = (
        root / "experiments/step3_gpt4.1mini_api_heuristic/score_outputs/misinfo_score_gpt-4.1-mini.csv"
    )
    llm_difficulty_path = (
        root / "experiments/step3_gpt4.1mini_api/difficulty_outputs/misinfo_difficulty_gpt-4.1-mini.csv"
    )
    heuristic_difficulty_path = (
        root
        / "experiments/step3_gpt4.1mini_api_heuristic/difficulty_outputs/misinfo_difficulty_gpt-4.1-mini.csv"
    )

    cot_frame = pd.read_csv(cot_path, usecols=["id", "text"]).iloc[:100].copy()

    variants = {
        "misinfo_llm_gpt41mini": (llm_score_path, llm_difficulty_path),
        "misinfo_heuristic_gpt41mini": (heuristic_score_path, heuristic_difficulty_path),
    }

    for variant_name, (score_path, difficulty_path) in variants.items():
        variant_frame = build_variant_frame(
            cot_frame=cot_frame,
            score_path=score_path,
            difficulty_path=difficulty_path,
        )
        variant_frame = assign_split(variant_frame, seed=args.seed)
        write_split_files(variant_frame, args.output_root / variant_name)


if __name__ == "__main__":
    main()
