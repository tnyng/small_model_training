"""Build combined annotation sheets: old missed round-3 tricky rows + new round-4 rows."""
from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path("/Users/yangtian/Documents/law/calib/RuleFollower")
ROUND3 = ROOT / "modernbert_uq" / "results" / "audit" / "stageB" / "round3"
ROUND4 = ROOT / "modernbert_uq" / "results" / "audit" / "stageB" / "round4_tricky"
OUT = ROUND4 / "combined"
OUT.mkdir(parents=True, exist_ok=True)

PEOPLE = [
    ("A1", "Yang"),
    ("A2", "Yu"),
    ("A3", "Alexander"),
    ("A4", "Jingwei"),
    ("A5", "Tobi"),
    ("A6", "Heejin"),
]


def load_round3_row_bank() -> dict[str, dict]:
    row_bank: dict[str, dict] = {}
    for path in sorted(ROUND3.glob("A*_blank.csv")):
        with path.open() as f:
            for row in csv.DictReader(f):
                row_id = row["row_id"]
                if row_id not in row_bank:
                    dataset = row_id.split("__")[0] if "__" in row_id else "overlap"
                    row_bank[row_id] = {
                        "row_id": row_id,
                        "combo": row["combo"],
                        "dataset": dataset,
                        "score_annotator": row["score_annotator"],
                        "claim": row["claim"],
                        "source": row["source"],
                        "claim_well_formed": "",
                        "entailment": "",
                        "score_correct": "",
                        "notes": "",
                    }
    return row_bank


def load_round3_missing() -> dict[str, list[str]]:
    with (ROUND3 / "assignment_key.csv").open() as f:
        rows = list(csv.DictReader(f))
    tricky = [row for row in rows if int(row["score_annotator"]) in (3, 7)]
    missing: dict[str, list[str]] = {key: [] for key, _ in PEOPLE if key != "A6"}
    for key in missing:
        for row in tricky:
            seen = row["annotators"].split("|")
            if key not in seen:
                missing[key].append(row["row_id"])
    return missing


def load_round4_rows() -> dict[str, list[dict]]:
    rows_by_person: dict[str, list[dict]] = {}
    for _, display_name in PEOPLE:
        path = ROUND4 / f"Annotation_{display_name}_blank.csv"
        with path.open() as f:
            rows_by_person[display_name] = list(csv.DictReader(f))
    return rows_by_person


def write_outputs() -> None:
    row_bank = load_round3_row_bank()
    missing = load_round3_missing()
    round4_rows = load_round4_rows()

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

    assignment_rows: list[dict] = []

    for key, display_name in PEOPLE:
        combined: list[dict] = []
        old_count = 0
        if key != "A6":
            for row_id in missing[key]:
                combined.append(dict(row_bank[row_id]))
                old_count += 1
                assignment_rows.append(
                    {
                        "annotator_key": key,
                        "annotator": display_name,
                        "row_id": row_id,
                        "pack": "round3_old_missing",
                    }
                )

        for row in round4_rows[display_name]:
            combined.append(row)
            assignment_rows.append(
                {
                    "annotator_key": key,
                    "annotator": display_name,
                    "row_id": row["row_id"],
                    "pack": "round4_new",
                }
            )

        out_path = OUT / f"Annotation_{display_name}_combined.csv"
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(combined)

    with (OUT / "combined_assignment_key.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["annotator_key", "annotator", "row_id", "pack"]
        )
        writer.writeheader()
        writer.writerows(assignment_rows)

    lines = [
        "# Combined Annotation Sheets",
        "",
        "- Original five annotators: old missed round-3 score-3/7 rows + round-4 new rows",
        "- Heejin: round-4 new rows only",
        "",
        "## Expected totals",
        "",
        "- Yang: 9 old + 21 new = 30",
        "- Yu: 8 old + 21 new = 29",
        "- Alexander: 11 old + 21 new = 32",
        "- Jingwei: 9 old + 21 new = 30",
        "- Tobi: 11 old + 21 new = 32",
        "- Heejin: 21 new = 21",
        "",
    ]
    (OUT / "README.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    write_outputs()
