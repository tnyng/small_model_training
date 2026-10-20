"""Matched binary-G versus score-G comparison on one frozen teacher split.

Both G models are evaluated against the *same* new G=5 endpoint. The original
binary teacher target is not substituted as a reference label for this test.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def load_predictions(path: Path) -> dict[str, dict]:
    result = {}
    for row in read_jsonl(path):
        uid = row["uid"]
        if uid in result:
            raise ValueError(f"Duplicate prediction: {uid}")
        result[uid] = row
    return result


def ranking(labels: list[int], scores: list[float]) -> tuple[dict, list[dict]]:
    if len(labels) != len(scores) or not labels:
        raise ValueError("Missing labelled predictions")
    if any(y not in (0, 1) for y in labels):
        raise ValueError("Ranking labels must be binary")
    npos = sum(labels)
    nneg = len(labels) - npos
    if not npos or not nneg:
        raise ValueError("Ranking requires both classes")
    ordered = sorted(zip(scores, labels), key=lambda pair: pair[0], reverse=True)
    tp = 0
    curve = []
    for i, (score, label) in enumerate(ordered, 1):
        tp += label
        if i <= 100 or i % 100 == 0 or i == len(ordered):
            curve.append({"n_triggered": i, "coverage": i / len(ordered),
                          "precision": tp / i, "recall": tp / npos,
                          "score_threshold": score})
    # Average precision at distinct score thresholds: all tied examples enter
    # together, so file/UID order cannot inflate AP.
    ap = 0.
    tp = 0
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and ordered[j][0] == ordered[i][0]:
            j += 1
        positives = sum(y for _, y in ordered[i:j])
        tp += positives
        ap += (tp / j) * positives
        i = j
    # Pairwise AUC with half-credit for exact score ties.
    asc = sorted(zip(scores, labels))
    negatives_below = 0
    wins = 0.
    i = 0
    while i < len(asc):
        j = i + 1
        while j < len(asc) and asc[j][0] == asc[i][0]:
            j += 1
        positives = sum(y for _, y in asc[i:j])
        negatives = j - i - positives
        wins += positives * (negatives_below + negatives / 2)
        negatives_below += negatives
        i = j
    result = {"n": len(labels), "positive": npos, "base_rate": npos / len(labels),
              "ap": ap / npos, "auc": wins / (npos * nneg)}
    for k in (20, 100, 500, 1000):
        if len(labels) >= k:
            result[f"precision_at_{k}"] = sum(y for _, y in ordered[:k]) / k
    return result, curve


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--score", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--partition", choices=("dev", "test"), required=True)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {args.out}")
    binary = load_predictions(args.binary)
    graded = load_predictions(args.score)
    sidecar = {r["uid"]: r for r in read_jsonl(args.sidecar)}
    if set(binary) != set(graded) or set(binary) != set(sidecar):
        raise ValueError("Predictions and score sidecar have different sentence IDs")
    labels = {"G5": [], "F": []}
    scores = {m: {h: [] for h in labels} for m in ("binary", "graded")}
    for uid in sorted(binary):
        old, new, label = binary[uid], graded[uid], sidecar[uid]
        if old["cot_id"] != new["cot_id"] or old["sid"] != new["sid"]:
            raise ValueError(f"CoT/sentence ID mismatch: {uid}")
        if old["source_text_sha256"] != new["source_text_sha256"]:
            raise ValueError(f"Source hash mismatch: {uid}")
        if old["labels"]["F"] != new["labels"]["F"] or old["masks"]["F"] != new["masks"]["F"]:
            raise ValueError(f"F teacher mismatch: {uid}")
        if bool(label["mask_G_score"]) != bool(new["masks"]["G"]):
            raise ValueError(f"G-score mask mismatch: {uid}")
        if label["mask_G_score"]:
            labels["G5"].append(int(label["G_score"] == 5))
            scores["binary"]["G5"].append(float(old["scores"]["G"]))
            scores["graded"]["G5"].append(float(new["scores"]["G5"]))
        if old["masks"]["F"]:
            labels["F"].append(int(old["labels"]["F"]))
            scores["binary"]["F"].append(float(old["scores"]["F"]))
            scores["graded"]["F"].append(float(new["scores"]["F"]))
    args.out.mkdir(parents=True)
    metrics = {}
    for head in labels:
        metrics[head] = {}
        for model in scores:
            metric, curve = ranking(labels[head], scores[model][head])
            metrics[head][model] = metric
            with (args.out / f"{head}_{model}_precision_coverage.csv").open(
                    "w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=curve[0].keys())
                writer.writeheader(); writer.writerows(curve)
        metrics[head]["graded_minus_binary"] = {
            key: metrics[head]["graded"][key] - metrics[head]["binary"][key]
            for key in ("ap", "auc", "precision_at_20", "precision_at_100")}
    report = {"partition": args.partition, "n_sentences": len(binary),
              "reference": "new Luna G_score=5 and unchanged F teacher labels",
              "metrics": metrics}
    (args.out / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
