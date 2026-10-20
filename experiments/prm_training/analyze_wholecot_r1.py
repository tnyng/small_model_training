"""Paired, source-disjoint dev analysis of E2a and whole-CoT PRM predictions."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load(path: Path) -> dict[str, dict]:
    rows = {}
    with path.open() as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                uid = row["uid"]
                if uid in rows:
                    raise ValueError(f"Duplicate uid in {path}: {uid}")
                rows[uid] = row
    return rows


def ap(ranked: list[dict], head: str) -> float | None:
    positives = sum(row["labels"][head] == 1 for row in ranked)
    if not positives:
        return None
    hits = 0
    area = 0.0
    for rank, row in enumerate(ranked, 1):
        if row["labels"][head] == 1:
            hits += 1
            area += hits / rank
    return area / positives


def summarize(rows: list[dict], head: str) -> dict:
    labelled = [row for row in rows if row["masks"][head]]
    ranked = sorted(labelled, key=lambda row: row["scores"][head], reverse=True)
    positives = sum(row["labels"][head] == 1 for row in ranked)
    out = {"n": len(ranked), "positives": positives,
           "base_rate": positives / len(ranked) if ranked else None,
           "ap": ap(ranked, head)}
    out["top_k"] = {}
    for k in (10, 25, 50, 100, 200, 500):
        if len(ranked) >= k:
            hits = sum(row["labels"][head] == 1 for row in ranked[:k])
            out["top_k"][str(k)] = {"precision": hits / k, "recall": hits / positives if positives else None}
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e2a", type=Path, required=True)
    parser.add_argument("--wholecot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    old, new = load(args.e2a), load(args.wholecot)
    if old.keys() != new.keys():
        raise ValueError(f"UID mismatch: E2a-only={len(old.keys()-new.keys())}, whole-CoT-only={len(new.keys()-old.keys())}")
    for uid in old:
        for head in ("G", "F"):
            if old[uid]["labels"][head] != new[uid]["labels"][head] or old[uid]["masks"][head] != new[uid]["masks"][head]:
                raise ValueError(f"Label/mask mismatch at {uid}/{head}")

    report: dict = {"paired_uids": len(new), "models": {}}
    for name, data in (("e2a", old), ("wholecot", new)):
        rows = list(data.values())
        groups: dict[str, list[dict]] = defaultdict(list)
        datasets: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            groups[row["generator"]].append(row)
            datasets[row["task"]].append(row)
        report["models"][name] = {
            "overall": {head: summarize(rows, head) for head in ("G", "F")},
            "generators": {group: {head: summarize(group_rows, head) for head in ("G", "F")}
                           for group, group_rows in sorted(groups.items())},
            "datasets": {dataset: {head: summarize(group_rows, head) for head in ("G", "F")}
                         for dataset, group_rows in sorted(datasets.items())},
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"paired_uids": report["paired_uids"],
                      "overall": {name: data["overall"] for name, data in report["models"].items()}}, indent=2))


if __name__ == "__main__":
    main()
