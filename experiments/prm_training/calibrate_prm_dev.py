"""Source-disjoint dev calibration and conservative reward-gate candidates.

This never reads the human test set. Calibration and threshold selection use
different exact-source groups within dev. The selected threshold is only a
development-set operating point; independent human validation is still needed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


def load(path: Path) -> list[dict]:
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def partition(row: dict) -> str:
    digest = hashlib.sha256(("prm-cal-v1:" + row["source_text_sha256"]).encode()).hexdigest()
    return "calibration" if int(digest[:8], 16) / 16**8 < 0.6 else "threshold"


def platt_fit(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    x = np.log(np.clip(p, 1e-7, 1 - 1e-7) / (1 - np.clip(p, 1e-7, 1 - 1e-7)))
    xt = torch.tensor(x, dtype=torch.float64)
    yt = torch.tensor(y, dtype=torch.float64)
    log_slope = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
    bias = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
    optimizer = torch.optim.LBFGS([log_slope, bias], lr=0.5, max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        logits = torch.exp(log_slope) * xt + bias
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, yt)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.exp(log_slope).detach()), float(bias.detach())


def calibrated(p: np.ndarray, slope: float, bias: float) -> np.ndarray:
    q = np.clip(p, 1e-7, 1 - 1e-7)
    logits = slope * np.log(q / (1 - q)) + bias
    return 1 / (1 + np.exp(-np.clip(logits, -50, 50)))


def quality(y: np.ndarray, p: np.ndarray) -> dict:
    q = np.clip(p, 1e-7, 1 - 1e-7)
    ece = 0.0
    for left, right in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        keep = (p >= left) & (p < right if right < 1 else p <= right)
        if keep.any():
            ece += keep.mean() * abs(float(p[keep].mean() - y[keep].mean()))
    return {"brier": float(np.mean((p - y) ** 2)),
            "log_loss": float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))),
            "ece_10_equal_width": float(ece)}


def gate(y: np.ndarray, p: np.ndarray, target: float, minimum: int) -> dict | None:
    order = np.argsort(-p, kind="stable")
    sorted_y, sorted_p = y[order], p[order]
    cumulative = np.cumsum(sorted_y)
    selected = None
    for index in range(len(sorted_y)):
        n = index + 1
        if n < minimum or (index + 1 < len(sorted_y) and sorted_p[index + 1] == sorted_p[index]):
            continue
        hits = int(cumulative[index])
        if hits / n >= target:
            selected = {"threshold": float(sorted_p[index]), "triggered": n,
                        "true_positive": hits, "false_positive": n - hits,
                        "precision_on_threshold_split": hits / n,
                        "recall_on_threshold_split": hits / int(y.sum()) if y.sum() else None,
                        "false_triggers_per_1000": (n - hits) * 1000 / len(y)}
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-triggers", type=int, default=20)
    args = parser.parse_args()
    rows = load(args.predictions)
    groups: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        groups[partition(row)].add(row["source_text_sha256"])
    if groups["calibration"] & groups["threshold"]:
        raise ValueError("Source group leakage between calibration and threshold partitions")
    report = {"predictions": str(args.predictions),
              "source_groups": {part: len(groups[part]) for part in groups},
              "minimum_triggers": args.min_triggers, "heads": {}}
    for head in ("G", "F"):
        arrays = {}
        for part in ("calibration", "threshold"):
            subset = [r for r in rows if partition(r) == part and r["masks"][head]]
            y = np.array([r["labels"][head] for r in subset], dtype=int)
            p = np.array([r["scores"][head] for r in subset], dtype=float)
            if not len(y) or not y.any() or y.all():
                raise ValueError(f"Insufficient {head} labels in {part}")
            arrays[part] = y, p
        slope, bias = platt_fit(*arrays["calibration"])
        y, raw = arrays["threshold"]
        adjusted = calibrated(raw, slope, bias)
        report["heads"][head] = {
            "calibration_fit": {"slope": slope, "bias": bias,
                                "n": len(arrays["calibration"][0]),
                                "positives": int(arrays["calibration"][0].sum())},
            "threshold_split": {"n": len(y), "positives": int(y.sum()),
                                "raw": quality(y, raw), "calibrated": quality(y, adjusted)},
            "candidate_gates": {str(target): gate(y, adjusted, target, args.min_triggers)
                                for target in (0.9, 0.95, 0.99)},
        }
    report["warning"] = "Threshold-split precision is used for selection and is optimistic; validate frozen gates on independent human labels. Null means no gate meets the target with the minimum support."
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
