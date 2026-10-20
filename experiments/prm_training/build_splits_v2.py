"""Build leakage-safe PRM splits from the validated, heldout-filtered release.

The split unit is the exact classification-input hash, not the dataset row ID.
Every generator and dataset sharing that input remains in one partition.
"""

from __future__ import annotations

import argparse
import collections
import csv
import gzip
import hashlib
import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PACKAGE = ROOT / "experiments/hf16_cross_model_full/proportional_2000/hf_upload_teacher_2k_v1"
DEFAULT_HUMAN = ROOT / "experiments/human_test_set_150/source_pool_160/source_manifest.csv"


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def jsonl_gz(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", type=Path, default=DEFAULT_PACKAGE)
    ap.add_argument("--human-manifest", type=Path, default=DEFAULT_HUMAN)
    ap.add_argument("--out", type=Path, default=ROOT / "experiments/prm_training/splits_v2")
    ap.add_argument("--seed", type=int, default=20261008)
    ap.add_argument("--dev-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    args = ap.parse_args()
    if not 0 < args.dev_frac < 1 or not 0 < args.test_frac < 1 or args.dev_frac + args.test_frac >= 1:
        ap.error("dev/test fractions must be positive and sum to less than 1")
    if args.out.exists() and any(args.out.iterdir()):
        ap.error(f"Refusing to overwrite nonempty output: {args.out}")

    human_hashes = {r["text_sha256"] for r in csv.DictReader(args.human_manifest.open(newline="", encoding="utf-8"))}
    cot_by_id: dict[str, dict] = {}
    hash_tasks: dict[str, set[str]] = collections.defaultdict(set)
    for path in sorted((args.package / "cots").glob("*.jsonl.gz")):
        generator = path.name.removesuffix(".jsonl.gz")
        for cot in jsonl_gz(path):
            cid = cot["cot_id"]
            if cid in cot_by_id:
                raise ValueError(f"duplicate cot_id: {cid}")
            source_text = cot["classification_input"]
            if not source_text or not cot.get("task_definition"):
                raise ValueError(f"missing task/source for {cid}")
            h = sha256(source_text)
            if h in human_hashes:
                raise ValueError(f"package contains held-out human input: {cid}")
            cot_by_id[cid] = {"record": cot, "source_hash": h, "generator": generator}
            hash_tasks[h].add(cot["task"])
    if not cot_by_id:
        raise ValueError("no CoTs found")

    # Stratify approximately by the lexicographically first task represented by
    # each exact-text group; cross-task duplicate texts still have one assignment.
    groups_by_task: dict[str, list[str]] = collections.defaultdict(list)
    for h, tasks in hash_tasks.items():
        groups_by_task[min(tasks)].append(h)
    rng = random.Random(args.seed)
    split_for_hash: dict[str, str] = {}
    for task in sorted(groups_by_task):
        hashes = sorted(groups_by_task[task])
        rng.shuffle(hashes)
        n_dev = max(1, round(len(hashes) * args.dev_frac))
        n_test = max(1, round(len(hashes) * args.test_frac))
        for i, h in enumerate(hashes):
            split_for_hash[h] = "dev" if i < n_dev else "test" if i < n_dev + n_test else "train"

    args.out.mkdir(parents=True, exist_ok=False)
    writers = {s: (args.out / f"{s}.jsonl").open("w", encoding="utf-8") for s in ("train", "dev", "test")}
    counts = collections.Counter()
    cot_seen = collections.Counter()
    try:
        for path in sorted((args.package / "sentence_labels").glob("*.jsonl.gz")):
            for label in jsonl_gz(path):
                cid = label["cot_id"]
                if cid not in cot_by_id:
                    raise ValueError(f"label without CoT: {cid}")
                entry = cot_by_id[cid]
                cot = entry["record"]
                sid = label["sid"]
                sentences = cot["sentence_rows"]
                index = next((i for i, row in enumerate(sentences) if row["sid"] == sid), None)
                if index is None or sentences[index]["text"] != label["sentence"]:
                    raise ValueError(f"sentence mismatch: {cid}/{sid}")
                prior = " ".join(row["text"] for row in sentences[max(0, index - 4):index])
                masks = label.get("training_masks") or {}
                rec = {
                    "uid": hashlib.sha1(f"{cid}|{sid}".encode()).hexdigest()[:16],
                    "source_key": f"{label['task']}::{label['source_id']}",
                    "source_text_sha256": entry["source_hash"],
                    "task": label["task"], "source_id": label["source_id"],
                    "cot_id": cid, "generator": entry["generator"], "sid": sid,
                    "sentence": label["sentence"], "prior_context_4": prior,
                    "G": label.get("G"), "F": label.get("F"),
                    "mask_G": bool(masks.get("G")) and label.get("G") is not None,
                    "mask_F": bool(masks.get("F")) and label.get("F") is not None,
                    "task_definition": cot["task_definition"],
                    "classification_input": cot["classification_input"],
                }
                split = split_for_hash[entry["source_hash"]]
                writers[split].write(json.dumps(rec, ensure_ascii=False) + "\n")
                counts[f"rows_{split}"] += 1
                counts[f"G1_{split}"] += rec["G"] == 1
                counts[f"F1_{split}"] += rec["F"] == 1
                cot_seen[(split, cid)] += 1
    finally:
        for writer in writers.values():
            writer.close()

    if set(cid for _, cid in cot_seen) != set(cot_by_id):
        raise ValueError("not every CoT has labels")
    sets = {s: {h for h, v in split_for_hash.items() if v == s} for s in ("train", "dev", "test")}
    if any(sets[a] & sets[b] for a, b in (("train", "dev"), ("train", "test"), ("dev", "test"))):
        raise AssertionError("text hash leakage between splits")
    if any(sets[s] & human_hashes for s in sets):
        raise AssertionError("human test leakage")
    summary = {
        "seed": args.seed, "package": str(args.package), "human_manifest": str(args.human_manifest),
        "split_unit": "exact classification_input SHA256 across all tasks/generators",
        "source_hash_groups": {s: len(sets[s]) for s in sets},
        "cot_count": len(cot_by_id), "sentence_count": sum(counts[f"rows_{s}"] for s in sets),
        "counts": dict(counts), "human_hash_count": len(human_hashes),
        "pairwise_hash_overlap": 0, "human_hash_overlap": 0,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
