#!/usr/bin/env python3
"""Summarize all 20 ModernBERT UQ experiment outputs into a single comparison CSV.

Walks outputs/{condition}/frac_{f}/ directories, loads metrics.json files,
and emits a comparison CSV with condition, fraction, and key metrics.

Usage:
    python scripts/summarize_experiment_suite.py \
        --outputs-root outputs \
        --split test
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

CONDITIONS = ["reg_base", "reg_ensemble", "bt_base", "bt_ensemble"]
FRACTIONS = [0.1, 0.25, 0.5, 0.75, 1.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize experiment suite outputs.")
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--split", type=str, default="test", choices=["test", "validation"])
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def frac_to_label(fraction: float) -> str:
    return f"frac_{fraction:.2f}"


def load_metrics(metrics_path: Path) -> dict[str, float]:
    if not metrics_path.exists():
        return {}
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    result: dict[str, float] = {}
    for section, values in payload.items():
        if not isinstance(values, dict):
            continue
        for k, v in values.items():
            if v is not None and not (isinstance(v, float) and math.isnan(float(v))):
                result[f"{section}_{k}"] = float(v)
            elif v is not None:
                result[f"{section}_{k}"] = float("nan")
    return result


def collect_rows(outputs_root: Path, split: str) -> list[dict[str, object]]:
    rows = []
    for condition in CONDITIONS:
        for fraction in FRACTIONS:
            frac_label = frac_to_label(fraction)
            exp_dir = outputs_root / condition / frac_label
            metrics_path = exp_dir / f"{split}_metrics.json"
            if not metrics_path.exists():
                continue
            metrics = load_metrics(metrics_path)
            row: dict[str, object] = {
                "condition": condition,
                "fraction": fraction,
                "exp_dir": str(exp_dir),
            }
            row.update(metrics)
            rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    rows = collect_rows(args.outputs_root.resolve(), args.split)
    if not rows:
        raise SystemExit(f"No completed outputs found under {args.outputs_root}")

    frame = pd.DataFrame(rows).sort_values(["condition", "fraction"])

    out_path = args.out or (args.outputs_root / f"experiment_suite_{args.split}_summary.csv")
    frame.to_csv(out_path, index=False)
    print(f"Wrote {len(frame)} rows to {out_path}")
    cols = ["condition", "fraction"]
    for col in ["score_regression_rmse", "score_regression_pearson", "difficulty_rmse", "difficulty_pearson"]:
        if col in frame.columns:
            cols.append(col)
    print(frame[cols].to_string())


if __name__ == "__main__":
    main()