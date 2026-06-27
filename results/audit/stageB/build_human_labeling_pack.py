"""Build a 30-row blind hand-labeling CSV for the user, plus a hidden answer-key
keyed by row_id. After hand-labeling, run merge_human_labels.py to compute
agreement per rubric field and per score bin.
"""
from __future__ import annotations
import csv, json, random
from pathlib import Path

csv.field_size_limit(2**31 - 1)

ROOT = Path("/Users/yangtian/Documents/law/calib/RuleFollower")
OUT = ROOT / "modernbert_uq" / "results" / "audit" / "stageB"

# Read all 90 rows + my labels
rows_all = list(csv.DictReader((OUT/"per_branch_labels.csv").open()))
sample_meta = {r["row_id"]: r for r in csv.DictReader((OUT/"per_branch_sample.csv").open())}

# Pull FULL source from Layer 2 directly using (combo, dataset, property) as key
EXP = ROOT / "experiments"
COMBO_FOLDER = {
    "heuristic_ann_gpt41mini":     "annotation_gpt4.1mini_heuristic",
    "llm_gpt5mini_ann_gpt41mini":  "annotation_gpt4.1mini_gpt-5-mini",
    "llm_gptoss_ann_gpt41mini":    "annotation_gpt4.1mini_gpt-oss",
    "heuristic":                   "annotation_gpt4.1mini_heuristic",
    "llm_gpt5mini":                "annotation_gpt4.1mini_gpt-5-mini",
    "llm_gptoss":                  "annotation_gpt4.1mini_gpt-oss",
}

# Build a (combo, dataset, property_text) -> full_source lookup
print("Loading full Layer 2 sources for join ...")
l2_lookup = {}  # (folder_key, dataset, property_norm) -> full_source
import re
def norm(s): return re.sub(r"\s+", " ", (s or "").strip().lower())

# We need to figure out which combo each row belongs to (combo column in labels).
needed_combos = set(r["combo"] for r in rows_all)
for combo_short in needed_combos:
    folder = COMBO_FOLDER[combo_short]
    base = EXP / folder / "accuracy_outputs"
    for f in base.glob("*.csv"):
        ds = f.stem.split("_score_")[0]
        for row in csv.DictReader(f.open()):
            l2_lookup[(combo_short, ds, norm(row.get("property","")))] = row.get("text","")

# Attach full source to each row
missing = 0
for r in rows_all:
    sm = sample_meta[r["row_id"]]
    r["property"] = sm["property"]
    full = l2_lookup.get((r["combo"], sm["dataset"], norm(sm["property"])), None)
    if full is None:
        missing += 1
        r["source"] = sm["source"]  # fallback to truncated
    else:
        r["source"] = full
print(f"  attached full source for {len(rows_all) - missing}/{len(rows_all)} rows ({missing} fallbacks)")

# Stratified 30-row sample: 10 per combo; within each combo 4 from 1-4, 2 from 5-7,
# 3 from 8-9, 1 from flag pool (the flag pool sits at indices 25..29 in our sample plan).
random.seed(20260504)

def bin_of(score_str):
    s = int(score_str)
    if 1 <= s <= 4: return "1-4"
    if 5 <= s <= 7: return "5-7"
    if 8 <= s <= 9: return "8-9"
    return "?"

picks = []
for combo in ("heuristic", "llm_gpt5mini", "llm_gptoss"):
    combo_rows = [r for r in rows_all if r["combo"] == combo]
    # Index in their per-combo order: rows 25-29 are the flag-pool stratum
    by_idx = {r["row_id"]: i for i, r in enumerate(combo_rows)}
    main = [r for r in combo_rows if by_idx[r["row_id"]] < 25]
    flag = [r for r in combo_rows if by_idx[r["row_id"]] >= 25]
    by_bin = {"1-4": [], "5-7": [], "8-9": []}
    for r in main:
        by_bin[bin_of(r["score"])].append(r)
    targets = {"1-4": 4, "5-7": 2, "8-9": 3}
    for b, k in targets.items():
        picks.extend(random.sample(by_bin[b], min(k, len(by_bin[b]))))
    picks.extend(random.sample(flag, min(1, len(flag))))

# Shuffle the order so the combos and bins don't stack
random.shuffle(picks)

# Blind CSV — strip my labels, keep claim/source visible
blind_fields = ["row_id", "combo", "score_annotator", "coverage", "claim", "source",
                "claim_well_formed", "entailment", "score_correct", "vagueness", "notes"]
with (OUT/"human_labels_blank.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=blind_fields)
    w.writeheader()
    for r in picks:
        w.writerow({
            "row_id": r["row_id"],
            "combo": r["combo"],
            "score_annotator": r["score"],
            "coverage": r["coverage"],
            "claim": r["property"],
            "source": r["source"],
            "claim_well_formed": "",
            "entailment": "",
            "score_correct": "",
            "vagueness": "",
            "notes": "",
        })

# Hidden answer key — my labels, keyed by row_id
key_fields = ["row_id", "combo", "score_annotator", "claim_well_formed_claude",
              "entailment_claude", "score_correct_claude", "vagueness_claude",
              "accepted_claude", "note_claude"]
with (OUT/"claude_labels_for_comparison.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=key_fields)
    w.writeheader()
    for r in picks:
        w.writerow({
            "row_id": r["row_id"],
            "combo": r["combo"],
            "score_annotator": r["score"],
            "claim_well_formed_claude": r["claim_well_formed"],
            "entailment_claude": r["entailment"],
            "score_correct_claude": r["score_correct"],
            "vagueness_claude": r["vagueness"],
            "accepted_claude": r["accepted"],
            "note_claude": r["note"],
        })

print(f"Wrote {OUT/'human_labels_blank.csv'} with {len(picks)} rows")
print(f"Wrote {OUT/'claude_labels_for_comparison.csv'} (hidden key)")
print()
print("Stratification of the 30 picks:")
from collections import Counter
c = Counter((r["combo"], bin_of(r["score"])) for r in picks)
for k, v in sorted(c.items()): print(f"  {k}: {v}")
