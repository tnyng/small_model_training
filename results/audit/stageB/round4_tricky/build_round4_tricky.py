"""Build round-4 tricky-score hand-labeling pack for 6 annotators.

Design:
  - 42 new unique rows total
  - exact scores only: 21 score-3 rows + 21 score-7 rows
  - 3 labels per row
  - 6 annotators, 21 rows each
  - source branches are the old step-3 filtered branch files only
  - exclude datasets: article_bias, bureaucracies, tweets_news, misinfo_cancer
  - exclude all row_ids already used in earlier human rounds
"""
from __future__ import annotations

import csv
import random
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path("/Users/yangtian/Documents/law/calib/RuleFollower")
DATA = ROOT / "modernbert_uq" / "data"
STAGE_B = ROOT / "modernbert_uq" / "results" / "audit" / "stageB"
ROUND3 = STAGE_B / "round3"
OUT = STAGE_B / "round4_tricky"
OUT.mkdir(parents=True, exist_ok=True)

ANNOTATORS = [
    ("A1", "Yang"),
    ("A2", "Yu"),
    ("A3", "Alexander"),
    ("A4", "Jingwei"),
    ("A5", "Tobi"),
    ("A6", "Heejin"),
]

COMBOS = [
    ("heuristic_ann_gpt41mini", "rulefollower_heuristic_ann_gpt41mini_reg_base__filtered"),
    ("llm_gpt5mini_ann_gpt41mini", "rulefollower_llm_gpt5mini_ann_gpt41mini_reg_base__filtered"),
    ("llm_gptoss_ann_gpt41mini", "rulefollower_llm_gptoss_ann_gpt41mini_ge8__filtered"),
]

EXCLUDE_DATASETS = {"article_bias", "bureaucracies", "tweets_news", "misinfo_cancer"}
TARGETS_PER_COMBO = {3: 7, 7: 7}  # 14 rows per combo, 42 total
DATASET_CAP_PER_COMBO_SCORE = 2


def load_excluded_row_ids() -> set[str]:
    ids: set[str] = set()
    csv_paths = [
        STAGE_B / "human_labels_blank_filled.csv",
        STAGE_B / "sample_label_yu.csv",
        STAGE_B / "per_branch_sample.csv",
        ROUND3 / "assignment_key.csv",
    ]
    for path in csv_paths:
        if not path.exists():
            continue
        for row in csv.DictReader(path.open()):
            row_id = row.get("row_id") or row.get("id") or ""
            if row_id:
                ids.add(row_id)
    return ids


def load_pool() -> list[dict]:
    excluded_ids = load_excluded_row_ids()
    rows: list[dict] = []
    for combo_name, branch_name in COMBOS:
        path = DATA / branch_name / "train.csv"
        for row in csv.DictReader(path.open()):
            row_id = row["id"]
            if row_id in excluded_ids:
                continue
            if row["dataset"] in EXCLUDE_DATASETS:
                continue
            score = int(round(float(row["accuracy_score"])))
            if score not in (3, 7):
                continue
            row["_combo"] = combo_name
            row["_score"] = score
            rows.append(row)
    return rows


def sample_rows(pool: list[dict], seed: int = 42) -> list[dict]:
    random.seed(seed)
    by_combo_score_dataset: dict[tuple[str, int, str], list[dict]] = defaultdict(list)
    for row in pool:
        key = (row["_combo"], row["_score"], row["dataset"])
        by_combo_score_dataset[key].append(row)

    chosen: list[dict] = []
    chosen_ids: set[str] = set()

    for combo_name, _ in COMBOS:
        for score, target in TARGETS_PER_COMBO.items():
            dataset_keys = [
                key for key in by_combo_score_dataset
                if key[0] == combo_name and key[1] == score
            ]
            random.shuffle(dataset_keys)
            selected_for_bucket = 0

            # First pass: spread across datasets with a light cap.
            for key in dataset_keys:
                if selected_for_bucket >= target:
                    break
                candidates = by_combo_score_dataset[key][:]
                random.shuffle(candidates)
                take = 0
                for row in candidates:
                    if row["id"] in chosen_ids:
                        continue
                    chosen.append(row)
                    chosen_ids.add(row["id"])
                    selected_for_bucket += 1
                    take += 1
                    if selected_for_bucket >= target or take >= DATASET_CAP_PER_COMBO_SCORE:
                        break

            # Second pass: fill any shortfall from remaining rows regardless of dataset.
            if selected_for_bucket < target:
                remaining = [
                    row
                    for (c, s, _), rows_here in by_combo_score_dataset.items()
                    if c == combo_name and s == score
                    for row in rows_here
                    if row["id"] not in chosen_ids
                ]
                random.shuffle(remaining)
                for row in remaining:
                    if selected_for_bucket >= target:
                        break
                    chosen.append(row)
                    chosen_ids.add(row["id"])
                    selected_for_bucket += 1

            if selected_for_bucket != target:
                raise RuntimeError(
                    f"Could not fill target for {combo_name} score={score}: "
                    f"got {selected_for_bucket}, wanted {target}"
                )

    if len(chosen) != 42:
        raise RuntimeError(f"Expected 42 chosen rows, got {len(chosen)}")
    return chosen


def assign_annotators(chosen: list[dict], seed: int = 43) -> list[tuple[dict, list[str]]]:
    random.seed(seed)
    chosen = chosen[:]
    random.shuffle(chosen)

    total_target = {key: 21 for key, _ in ANNOTATORS}
    score_target = {
        "A1": {3: 11, 7: 10},
        "A2": {3: 11, 7: 10},
        "A3": {3: 11, 7: 10},
        "A4": {3: 10, 7: 11},
        "A5": {3: 10, 7: 11},
        "A6": {3: 10, 7: 11},
    }

    assignments: list[tuple[dict, list[str]]] = []
    for row in chosen:
        score = row["_score"]
        scored: list[tuple[float, str]] = []
        for key, _ in ANNOTATORS:
            if total_target[key] <= 0:
                continue
            score_need = score_target[key][score]
            priority = score_need + (total_target[key] / 100.0) + random.random() * 0.001
            scored.append((priority, key))
        scored.sort(reverse=True)
        selected = [key for _, key in scored[:3]]
        if len(selected) != 3:
            raise RuntimeError(f"Could not assign 3 annotators for {row['id']}")
        for key in selected:
            total_target[key] -= 1
            score_target[key][score] -= 1
        assignments.append((row, selected))

    if any(v != 0 for v in total_target.values()):
        raise RuntimeError(f"Unbalanced total targets remain: {total_target}")
    return assignments


def write_outputs(assignments: list[tuple[dict, list[str]]]) -> None:
    by_annotator: dict[str, list[dict]] = defaultdict(list)
    key_rows: list[dict] = []

    for row, annotators in assignments:
        base = {
            "row_id": row["id"],
            "combo": row["_combo"],
            "dataset": row["dataset"],
            "score_annotator": row["_score"],
            "claim": row["property"],
            "source": row["source_text"],
            "claim_well_formed": "",
            "entailment": "",
            "score_correct": "",
            "notes": "",
        }
        for annotator in annotators:
            by_annotator[annotator].append(dict(base))
        key_rows.append(
            {
                "row_id": row["id"],
                "combo": row["_combo"],
                "dataset": row["dataset"],
                "score_annotator": row["_score"],
                "annotators": "|".join(annotators),
                "n_labels_target": "3",
            }
        )

    fields = [
        "row_id",
        "combo",
        "dataset",
        "score_annotator",
        "claim",
        "source",
        "claim_well_formed",
        "entailment",
        "score_correct",
        "notes",
    ]

    for key, display_name in ANNOTATORS:
        rows = by_annotator[key]
        random.shuffle(rows)
        out_path = OUT / f"Annotation_{display_name}_blank.csv"
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    key_fields = list(key_rows[0].keys())
    with (OUT / "assignment_key.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=key_fields)
        writer.writeheader()
        writer.writerows(key_rows)


def write_readme(assignments: list[tuple[dict, list[str]]]) -> None:
    rows = [row for row, _ in assignments]
    score_counts = Counter(row["_score"] for row in rows)
    dataset_counts = Counter(row["dataset"] for row in rows)
    combo_counts = Counter(row["_combo"] for row in rows)

    lines = [
        "# Round 4 Tricky-Score Audit Pack",
        "",
        "- 42 new unique rows",
        "- exact scores only: 21 rows with score 3, 21 rows with score 7",
        "- 3 annotators per row",
        "- 6 annotators, 21 rows each",
        "- source data uses old step-3 branch outputs only",
        "- excluded datasets: article_bias, bureaucracies, tweets_news, misinfo_cancer",
        "",
        "## Per-combo counts",
        "",
    ]
    for combo_name, count in sorted(combo_counts.items()):
        lines.append(f"- `{combo_name}`: {count}")
    lines.extend(["", "## Per-dataset counts", ""])
    for dataset, count in sorted(dataset_counts.items(), key=lambda x: (-x[1], x[0])):
        lines.append(f"- `{dataset}`: {count}")
    lines.extend(["", "## Score counts", ""])
    for score in (3, 7):
        lines.append(f"- score `{score}`: {score_counts[score]}")

    (OUT / "README.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    pool = load_pool()
    chosen = sample_rows(pool)
    assignments = assign_annotators(chosen)
    write_outputs(assignments)
    write_readme(assignments)


if __name__ == "__main__":
    main()
