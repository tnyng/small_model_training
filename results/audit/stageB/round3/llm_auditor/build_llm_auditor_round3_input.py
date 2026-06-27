from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd


ROOT = Path("/Users/yangtian/Documents/law/calib/RuleFollower/modernbert_uq/results/audit/stageB/round3")
OUT = ROOT / "llm_auditor"

INPUT_FILES = [ROOT / f"A{i}_blank.csv" for i in range(1, 6)]
BASE_FIELDS = [
    "row_id",
    "combo",
    "score_annotator",
    "claim",
    "source",
    "claim_well_formed",
    "entailment",
    "score_correct",
    "notes",
]
META_FIELDS = [
    "n_human_assignments",
    "annotator_packets",
]


def load_packets() -> pd.DataFrame:
    frames = []
    for path in INPUT_FILES:
        df = pd.read_csv(path, keep_default_na=False)
        df["annotator_packet"] = path.name
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def build_unique_rows(df: pd.DataFrame) -> pd.DataFrame:
    meta = (
        df.groupby("row_id", sort=False)
        .agg(
            n_human_assignments=("annotator_packet", "size"),
            annotator_packets=("annotator_packet", lambda xs: "|".join(sorted(set(xs)))),
        )
        .reset_index()
    )

    unique_rows = df.drop_duplicates(subset=["row_id"], keep="first").copy()
    for field in ("claim_well_formed", "entailment", "score_correct", "notes"):
        unique_rows[field] = ""

    unique_rows = unique_rows[BASE_FIELDS].merge(meta, on="row_id", how="left")
    return unique_rows


def main() -> None:
    csv.field_size_limit(2**31 - 1)
    OUT.mkdir(parents=True, exist_ok=True)

    all_rows = load_packets()
    unique_rows = build_unique_rows(all_rows)

    blank_path = OUT / "round3_unique_blank.csv"
    meta_path = OUT / "round3_unique_with_metadata.csv"

    unique_rows[BASE_FIELDS].to_csv(blank_path, index=False)
    unique_rows[BASE_FIELDS + META_FIELDS].to_csv(meta_path, index=False)

    print(f"total packet rows: {len(all_rows)}")
    print(f"unique claims: {len(unique_rows)}")
    print(f"wrote: {blank_path}")
    print(f"wrote: {meta_path}")


if __name__ == "__main__":
    main()
