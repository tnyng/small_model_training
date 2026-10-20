"""Compare the new whole-CoT PRM with E2a on identical dev sentence IDs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from train_prm import auc_ap


def read_predictions(path: Path) -> dict[str, dict]:
    with path.open() as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    result = {r["uid"]: r for r in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate uid in {path}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--e2a", type=Path, required=True)
    parser.add_argument("--wholecot", type=Path, required=True)
    args = parser.parse_args()
    old = read_predictions(args.e2a)
    new = read_predictions(args.wholecot)
    common = old.keys() & new.keys()
    print(f"E2a rows={len(old):,}; whole-CoT rows={len(new):,}; matched={len(common):,}")
    for head in ("G", "F"):
        included = sorted(uid for uid in common if old[uid]["masks"][head] and new[uid]["masks"][head])
        y_old = np.array([old[uid]["labels"][head] for uid in included], dtype=int)
        y_new = np.array([new[uid]["labels"][head] for uid in included], dtype=int)
        if not np.array_equal(y_old, y_new):
            raise ValueError(f"{head} labels differ between runs")
        for name, rows in (("E2a", old), ("whole-CoT", new)):
            scores = np.array([rows[uid]["scores"][head] for uid in included])
            auc, ap = auc_ap(y_old, scores)
            print(f"{head} {name}: n={len(included):,} positive={int(y_old.sum()):,} "
                  f"AUROC={auc:.5f} AP={ap:.5f}")


if __name__ == "__main__":
    main()
