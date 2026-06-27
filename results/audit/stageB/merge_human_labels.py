"""Merge user's hand-labels (in human_labels_filled.csv) with my labels
(claude_labels_for_comparison.csv). Compute per-field agreement,
per-score-bin agreement, and a disagreement summary.

Run AFTER you finish hand-labeling human_labels_blank.csv and rename it to
human_labels_filled.csv (or pass a different path as argv[1]).
"""
from __future__ import annotations
import csv, sys
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path("/Users/yangtian/Documents/law/calib/RuleFollower")
OUT = ROOT / "modernbert_uq" / "results" / "audit" / "stageB"

human_path = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT / "human_labels_filled.csv"
key_path   = OUT / "claude_labels_for_comparison.csv"
if not human_path.exists():
    print(f"Could not find {human_path}.")
    print("After you finish hand-labeling, save your CSV as human_labels_filled.csv in this folder, then re-run.")
    sys.exit(1)

human = {r["row_id"]: r for r in csv.DictReader(human_path.open())}
key   = {r["row_id"]: r for r in csv.DictReader(key_path.open())}

# Ensure same set of row_ids
common = sorted(set(human) & set(key))
missing = sorted(set(key) - set(human))
if missing:
    print(f"NOTE: {len(missing)} row_ids in key but not in human file: {missing[:5]}{'...' if len(missing)>5 else ''}")
print(f"Comparing {len(common)} row_ids.")

def bin_of(score_str):
    s = int(score_str)
    if 1 <= s <= 4: return "1-4"
    if 5 <= s <= 7: return "5-7"
    if 8 <= s <= 9: return "8-9"
    return "?"

# Cohen's kappa for categorical labels
def cohens_kappa(a_labels, b_labels):
    n = len(a_labels)
    if n == 0: return None, 0
    cats = sorted(set(a_labels) | set(b_labels))
    if len(cats) < 2: return 1.0, n
    po = sum(1 for x,y in zip(a_labels, b_labels) if x==y) / n
    pe = sum((a_labels.count(c)/n) * (b_labels.count(c)/n) for c in cats)
    if pe == 1: return 1.0, n
    return (po - pe) / (1 - pe), n

# Per-field agreement
fields = ["claim_well_formed", "entailment", "score_correct"]
key_field_map = {
    "claim_well_formed": "claim_well_formed_claude",
    "entailment": "entailment_claude",
    "score_correct": "score_correct_claude",
}

def cell(d, k):
    v = (d.get(k, "") or "").strip().lower()
    if v in ("", "n/a", "na"): return None
    return v

print("\n=== Per-field agreement (Cohen's kappa) ===")
for f in fields:
    ah, ak = [], []
    for rid in common:
        h = cell(human[rid], f)
        k = cell(key[rid], key_field_map[f])
        if h is None or k is None:
            continue
        ah.append(h); ak.append(k)
    if not ah:
        print(f"  {f}: no comparable rows")
        continue
    kappa, n = cohens_kappa(ah, ak)
    agree = sum(1 for x,y in zip(ah,ak) if x==y)
    print(f"  {f}: kappa = {kappa:.3f}   raw agreement = {agree}/{n} = {100*agree/n:.0f}%")

# Per-score-bin per-field agreement
print("\n=== Agreement by score bin ===")
for bin_name in ("1-4", "5-7", "8-9"):
    print(f"  bin {bin_name}:")
    for f in fields:
        ah, ak = [], []
        for rid in common:
            if bin_of(human[rid]["score_annotator"]) != bin_name: continue
            h = cell(human[rid], f); k = cell(key[rid], key_field_map[f])
            if h is None or k is None: continue
            ah.append(h); ak.append(k)
        if not ah:
            print(f"    {f}: no comparable rows")
            continue
        kappa, n = cohens_kappa(ah, ak)
        agree = sum(1 for x,y in zip(ah,ak) if x==y)
        print(f"    {f}: kappa = {kappa:.3f}   raw {agree}/{n} = {100*agree/n:.0f}%")

# Disagreement detail
print("\n=== Disagreements per row (first 15) ===")
out_rows = []
for rid in common:
    h = human[rid]; k = key[rid]
    diffs = []
    for f in fields:
        hv = cell(h, f); kv = cell(k, key_field_map[f])
        if hv is None or kv is None: continue
        if hv != kv: diffs.append(f"{f}: human={hv} claude={kv}")
    if diffs:
        out_rows.append({
            "row_id": rid, "combo": h["combo"], "score": h["score_annotator"],
            "diffs": " | ".join(diffs),
            "human_notes": h.get("notes", ""),
            "claude_notes": k.get("note_claude", ""),
        })

for r in out_rows[:15]:
    print(f"  [{r['row_id']} {r['combo']} score={r['score']}] {r['diffs']}")
    if r["human_notes"]: print(f"     your note: {r['human_notes']}")
    if r["claude_notes"]: print(f"     claude note: {r['claude_notes']}")

# Save full disagreement CSV
import csv as _csv
with (OUT/"human_vs_claude_disagreements.csv").open("w", newline="") as f:
    w = _csv.DictWriter(f, fieldnames=["row_id","combo","score","diffs","human_notes","claude_notes"])
    w.writeheader()
    for r in out_rows: w.writerow(r)
print(f"\nFull disagreements written to {OUT/'human_vs_claude_disagreements.csv'}")
print(f"Total disagreements: {len(out_rows)} of {len(common)} rows")
