from __future__ import annotations
import argparse, collections, csv, glob, hashlib, json, random
from pathlib import Path

LABELS = "experiments/hf16_cross_model_full/proportional_2000/luna_unified_v1/*/labels/*.jsonl"
CANON = "experiments/hf16_cross_model_full/proportional_2000/canonical/*/*.jsonl"
HELDOUT = "experiments/human_test_set_150/source_pool_160/source_manifest.csv"


def key(task: str, source_id: str) -> str:
    return f"{task}::{source_id}"


def load_labels() -> list[dict]:
    rows = []
    for path in sorted(glob.glob(LABELS)):
        generator = Path(path).parts[-3]
        for line in open(path):
            if not line.strip():
                continue
            r = json.loads(line)
            r["generator"] = generator
            rows.append(r)
    return rows


def load_sources() -> dict[str, dict]:
    out = {}
    for path in sorted(glob.glob(CANON)):
        for line in open(path):
            if not line.strip():
                continue
            r = json.loads(line)
            k = key(r["task"], r["source_id"])
            if k not in out:
                out[k] = {
                    "task_definition": r.get("task_definition", ""),
                    "classification_input": r.get("classification_input", ""),
                }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("experiments/prm_training/splits"))
    ap.add_argument("--seed", type=int, default=20261003)
    ap.add_argument("--dev-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    args = ap.parse_args()

    heldout = {
        key(r["dataset"], r["id"]) for r in csv.DictReader(open(HELDOUT))
    }
    rows = load_labels()
    sources = load_sources()

    by_task = collections.defaultdict(set)
    for r in rows:
        k = key(r["task"], r["source_id"])
        if k in heldout:
            continue
        by_task[r["task"]].add(k)

    rng = random.Random(args.seed)
    assign: dict[str, str] = {}
    for task in sorted(by_task):
        keys = sorted(by_task[task])
        rng.shuffle(keys)
        n = len(keys)
        n_dev = max(1, round(n * args.dev_frac))
        n_test = max(1, round(n * args.test_frac))
        for i, k in enumerate(keys):
            assign[k] = "dev" if i < n_dev else "test" if i < n_dev + n_test else "train"

    train_keys = sorted(k for k, v in assign.items() if v == "train")
    rng.shuffle(train_keys)
    ladder = [175, 350, 700, len(train_keys)]
    tiers = {n: set(train_keys[:n]) for n in ladder}

    args.out.mkdir(parents=True, exist_ok=True)
    kept = dropped = 0
    writers = {s: open(args.out / f"{s}.jsonl", "w") for s in ("train", "dev", "test")}
    for r in rows:
        k = key(r["task"], r["source_id"])
        split = assign.get(k)
        if split is None:
            dropped += 1
            continue
        masks = r.get("training_masks") or {}
        rec = {
            "uid": hashlib.sha1(f"{r['cot_id']}|{r['sid']}".encode()).hexdigest()[:16],
            "source_key": k,
            "task": r["task"],
            "source_id": r["source_id"],
            "cot_id": r["cot_id"],
            "generator": r["generator"],
            "sid": r["sid"],
            "sentence": r["sentence"],
            "G": r.get("G"),
            "F": r.get("F"),
            "mask_G": bool(masks.get("G")) and r.get("G") is not None,
            "mask_F": bool(masks.get("F")) and r.get("F") is not None,
            "task_definition": sources.get(k, {}).get("task_definition", ""),
            "classification_input": sources.get(k, {}).get("classification_input", ""),
            "tier": min((n for n in ladder if k in tiers[n]), default=None) if split == "train" else None,
        }
        writers[split].write(json.dumps(rec, ensure_ascii=False) + "\n")
        kept += 1
    for w in writers.values():
        w.close()

    summary = {
        "seed": args.seed,
        "split_key": "(task, source_id)",
        "heldout_excluded": len(heldout),
        "rows_kept": kept,
        "rows_dropped_heldout": dropped,
        "sources": {s: sum(1 for v in assign.values() if v == s) for s in ("train", "dev", "test")},
        "train_tiers": ladder,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
