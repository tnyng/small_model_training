"""Join frozen Luna G 1–5 labels to the existing leakage-safe PRM splits.

The output is a small sidecar, not a replacement for the binary G/F splits.
Every join is checked against the exact sentence text; technical nulls stay null.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPLITS = ROOT / "experiments/prm_training/splits_v2"
DEFAULT_SCORES = (ROOT / "experiments/hf16_cross_model_full/proportional_2000"
                  / "luna_gscore_v1")
DEFAULT_HUMAN = (ROOT / "experiments/human_test_set_150/source_pool_160"
                 / "source_manifest.csv")
DEFAULT_OUT = ROOT / "experiments/prm_training/gscore_sidecar_v1"
PARTITIONS = ("train", "dev", "test")


def lines(path: Path):
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{number}") from exc


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_scores(score_root: Path) -> tuple[dict[tuple[str, str], dict], dict]:
    scores = {}
    metadata = {"files": 0, "cots": 0, "sentences": 0,
                "prompt_sha256": set(), "teacher_model": set()}
    for path in sorted(score_root.glob("*/*/sentence_g_scores.jsonl")):
        summary = json.loads((path.parent / "summary.json").read_text(encoding="utf-8"))
        if summary.get("calls_failed") != 0:
            raise ValueError(f"Failed score calls in {path.parent}")
        metadata["files"] += 1
        metadata["cots"] += summary["cots"]
        metadata["prompt_sha256"].add(summary["prompt_sha256"])
        metadata["teacher_model"].add(summary["model"])
        seen_cots = set()
        seen_rows = 0
        for row in lines(path):
            key = (row["cot_id"], row["sid"])
            if key in scores:
                raise ValueError(f"Duplicate score row: {key}")
            score = row["G_score"]
            if score is not None and (type(score) is not int or score not in range(1, 6)):
                raise ValueError(f"Invalid G score: {key}={score!r}")
            if not isinstance(row.get("sentence"), str):
                raise ValueError(f"Missing sentence: {key}")
            scores[key] = {"sentence": row["sentence"], "G_score": score}
            seen_cots.add(row["cot_id"])
            seen_rows += 1
        if len(seen_cots) != summary["cots"] or seen_rows != summary["sentences"]:
            raise ValueError(f"Score file count mismatch: {path}")
        metadata["sentences"] += seen_rows
    if metadata["files"] != 64 or metadata["cots"] != 8000:
        raise ValueError(f"Expected 64 files/8,000 CoTs: {metadata}")
    if len(metadata["prompt_sha256"]) != 1 or len(metadata["teacher_model"]) != 1:
        raise ValueError("Mixed score prompt or teacher model")
    metadata["prompt_sha256"] = metadata["prompt_sha256"].pop()
    metadata["teacher_model"] = metadata["teacher_model"].pop()
    return scores, metadata


def build(splits: Path, score_root: Path, human_manifest: Path, out: Path) -> dict:
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    scores, score_meta = load_scores(score_root)
    human_hashes = {r["text_sha256"] for r in csv.DictReader(
        human_manifest.open(newline="", encoding="utf-8"))}
    counts = Counter()
    grade_counts = defaultdict(Counter)
    old_vs_new = Counter()
    task_generator = defaultdict(Counter)
    split_hashes = {partition: set() for partition in PARTITIONS}
    used_keys = set()
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="gscore-sidecar-", dir=out.parent) as temp:
        temp_dir = Path(temp)
        for partition in PARTITIONS:
            with (temp_dir / f"{partition}.jsonl").open("w", encoding="utf-8") as writer:
                for row in lines(splits / f"{partition}.jsonl"):
                    key = (row["cot_id"], row["sid"])
                    if key in used_keys:
                        raise ValueError(f"Duplicated split sentence: {key}")
                    used_keys.add(key)
                    label = scores.get(key)
                    if label is None or label["sentence"] != row["sentence"]:
                        raise ValueError(f"Missing or mismatched G score: {key}")
                    h = row["source_text_sha256"]
                    if h in human_hashes:
                        raise ValueError(f"Human source overlap: {key}")
                    split_hashes[partition].add(h)
                    score = label["G_score"]
                    sidecar = {"uid": row["uid"], "cot_id": key[0], "sid": key[1],
                               "sentence_sha256": sha256(row["sentence"]),
                               "G_score": score, "mask_G_score": score is not None}
                    writer.write(json.dumps(sidecar, ensure_ascii=False) + "\n")
                    counts[f"rows_{partition}"] += 1
                    grade_counts[partition]["null" if score is None else str(score)] += 1
                    old_vs_new[(partition, str(row["G"]),
                                "null" if score is None else str(score))] += 1
                    task_generator[(partition, row["task"], row["generator"])][
                        "null" if score is None else str(score)] += 1
        for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
            if split_hashes[left] & split_hashes[right]:
                raise ValueError(f"Source-hash overlap: {left}/{right}")
        if len(used_keys) != 251870:
            raise ValueError(f"Unexpected split sentence count: {len(used_keys)}")
        unused_cots = {cid for cid, _ in scores.keys() - used_keys}
        summary = {
            "source_splits": str(splits), "score_root": str(score_root),
            "human_manifest": str(human_manifest),
            "score_teacher": score_meta,
            "split_cots": len({cid for cid, _ in used_keys}),
            "split_sentences": len(used_keys), "unused_score_cots": len(unused_cots),
            "source_hash_groups": {p: len(split_hashes[p]) for p in PARTITIONS},
            "human_hash_overlap": 0, "cross_split_hash_overlap": 0,
            "rows": dict(counts),
            "G_score_counts": {p: dict(grade_counts[p]) for p in PARTITIONS},
            "old_binary_G_by_new_score": [
                {"split": p, "old_G": old, "G_score": score, "n": n}
                for (p, old, score), n in sorted(old_vs_new.items())],
            "by_task_generator": [
                {"split": p, "task": task, "generator": generator,
                 "G_score_counts": dict(values)}
                for (p, task, generator), values in sorted(task_generator.items())],
        }
        (temp_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n",
                                               encoding="utf-8")
        os.replace(temp_dir, out)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    parser.add_argument("--human-manifest", type=Path, default=DEFAULT_HUMAN)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    summary = build(args.splits, args.scores, args.human_manifest, args.out)
    print(json.dumps({k: summary[k] for k in (
        "split_cots", "split_sentences", "unused_score_cots", "G_score_counts",
        "human_hash_overlap", "cross_split_hash_overlap")}, indent=2))


if __name__ == "__main__":
    main()
