"""Build a 30-row blind hand-labeling pack for a SECOND annotator.

Composition:
- 25 NEW rows drawn from the 60 not yet user-labeled, stratified by combo and
  score bin.
- 5 OVERLAP rows drawn from the disagreement file (rows where Claude and the
  first annotator disagreed). These let us measure inter-annotator agreement
  on contested cases.

Outputs:
- human_labels_v2_blank.csv         — blind sheet for the second annotator
- claude_labels_v2_for_comparison.csv — Claude's labels (answer key)
- user_labels_v2_overlap.csv         — first annotator's labels on the 5 overlap rows
"""
from __future__ import annotations
import csv, random
from pathlib import Path
from collections import defaultdict

csv.field_size_limit(2**31 - 1)
ROOT = Path("/Users/yangtian/Documents/law/calib/RuleFollower")
OUT = ROOT / "modernbert_uq" / "results" / "audit" / "stageB"

claude_all = {r["row_id"]: r for r in csv.DictReader((OUT/"per_branch_labels.csv").open())}
user_30   = {r["row_id"]: r for r in csv.DictReader((OUT/"human_labels_blank_filled.csv").open())}
sample    = {r["row_id"]: r for r in csv.DictReader((OUT/"per_branch_sample.csv").open())}
disagree  = list(csv.DictReader((OUT/"human_vs_claude_disagreements.csv").open()))

# Layer 2 full-source lookup (same as v1 builder)
EXP = ROOT / "experiments"
COMBO_FOLDER = {
    "heuristic":     "annotation_gpt4.1mini_heuristic",
    "llm_gpt5mini":  "annotation_gpt4.1mini_gpt-5-mini",
    "llm_gptoss":    "annotation_gpt4.1mini_gpt-oss",
}
import re
def norm(s): return re.sub(r"\s+", " ", (s or "").strip().lower())
print("Loading Layer 2 sources …")
l2_lookup = {}
needed_combos = set(r["combo"] for r in claude_all.values())
for combo in needed_combos:
    folder = EXP / COMBO_FOLDER[combo] / "accuracy_outputs"
    for f in folder.glob("*.csv"):
        ds = f.stem.split("_score_")[0]
        for row in csv.DictReader(f.open()):
            l2_lookup[(combo, ds, norm(row.get("property","")))] = row.get("text","")

def bin_of(s):
    n = int(s);
    return "1-4" if n<=4 else ("5-7" if n<=7 else "8-9")

# ---------- 5 OVERLAP rows from disagreements ----------
random.seed(20260504)
# Ensure overlap is balanced across combos if possible: try 2/2/1
disagree_by_combo = defaultdict(list)
for d in disagree:
    disagree_by_combo[d["combo"]].append(d)
overlap_target = {"heuristic": 2, "llm_gpt5mini": 2, "llm_gptoss": 1}
overlap_picks = []
for c, k in overlap_target.items():
    pool = disagree_by_combo.get(c, [])
    if pool: overlap_picks.extend(random.sample(pool, min(k, len(pool))))
# Top up to 5 if any combo lacked disagreements
needed = 5 - len(overlap_picks)
remaining_disagree = [d for d in disagree if d["row_id"] not in {p["row_id"] for p in overlap_picks}]
if needed > 0 and remaining_disagree:
    overlap_picks.extend(random.sample(remaining_disagree, min(needed, len(remaining_disagree))))
overlap_ids = {p["row_id"] for p in overlap_picks}
print(f"Overlap rows: {len(overlap_ids)} from {len(disagree)} disagreement rows")

# ---------- 25 NEW rows from the 60 the user did NOT label ----------
unlabeled_ids = [rid for rid in claude_all if rid not in user_30]
unlabeled_rows = [claude_all[rid] for rid in unlabeled_ids]
# Stratify by combo + bin
by_combo_bin = defaultdict(list)
for r in unlabeled_rows:
    by_combo_bin[(r["combo"], bin_of(r["score"]))].append(r)
# Target shape per combo: 4 from 1-4, 2 from 5-7, 3 from 8-9 = 9 per combo. Total 27. Trim to 25.
new_picks = []
target_per_combo = [("1-4",4),("5-7",2),("8-9",3)]
for combo in ("heuristic","llm_gpt5mini","llm_gptoss"):
    for b, k in target_per_combo:
        pool = by_combo_bin.get((combo,b), [])
        if pool: new_picks.extend(random.sample(pool, min(k, len(pool))))
# Trim to 25
random.shuffle(new_picks)
new_picks = new_picks[:25]
print(f"New rows: {len(new_picks)} drawn from {len(unlabeled_rows)} unlabeled")

# ---------- Combine + write blind CSV ----------
all_picks = list(overlap_picks) + new_picks
random.shuffle(all_picks)

def get_full_source(combo, dataset, prop):
    return l2_lookup.get((combo, dataset, norm(prop)), "")

blind_fields = ["row_id", "combo", "score_annotator", "coverage", "claim", "source",
                "claim_well_formed", "entailment", "score_correct", "vagueness", "notes"]
with (OUT/"human_labels_v2_blank.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=blind_fields)
    w.writeheader()
    for r in all_picks:
        rid = r["row_id"]
        sm = sample[rid]
        full_src = get_full_source(r["combo"], sm["dataset"], sm["property"]) or sm["source"]
        w.writerow({
            "row_id": rid, "combo": r["combo"],
            "score_annotator": r["score"] if "score" in r else r.get("score",""),
            "coverage": claude_all[rid]["coverage"],
            "claim": sm["property"],
            "source": full_src,
            "claim_well_formed":"", "entailment":"", "score_correct":"", "vagueness":"", "notes":"",
        })

# ---------- Answer key (Claude) ----------
key_fields = ["row_id","combo","score_annotator","is_overlap",
              "claim_well_formed_claude","entailment_claude","score_correct_claude",
              "vagueness_claude","accepted_claude","note_claude"]
with (OUT/"claude_labels_v2_for_comparison.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=key_fields)
    w.writeheader()
    for r in all_picks:
        rid = r["row_id"]
        c = claude_all[rid]
        w.writerow({
            "row_id": rid, "combo": c["combo"], "score_annotator": c["score"],
            "is_overlap": "1" if rid in overlap_ids else "",
            "claim_well_formed_claude": c["claim_well_formed"],
            "entailment_claude": c["entailment"],
            "score_correct_claude": c["score_correct"],
            "vagueness_claude": c["vagueness"],
            "accepted_claude": c["accepted"],
            "note_claude": c["note"],
        })

# ---------- First-annotator's overlap labels (for kappa) ----------
ov_fields = ["row_id","combo","score_annotator",
             "claim_well_formed_user","entailment_user","score_correct_user",
             "vagueness_user","notes_user"]
with (OUT/"user_labels_v2_overlap.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=ov_fields)
    w.writeheader()
    for rid in overlap_ids:
        u = user_30[rid]
        w.writerow({
            "row_id": rid, "combo": u["combo"], "score_annotator": u["score_annotator"],
            "claim_well_formed_user": u["claim_well_formed"],
            "entailment_user": u["entailment"],
            "score_correct_user": u["score_correct"],
            "vagueness_user": u["vagueness"],
            "notes_user": u.get("notes",""),
        })

# Summary
print("\nStratification of v2 picks:")
strat = defaultdict(int)
for r in all_picks:
    strat[(r["combo"], bin_of(r["score"]), "overlap" if r["row_id"] in overlap_ids else "new")] += 1
for k,v in sorted(strat.items()): print(f"  {k}: {v}")
print(f"\nWrote:")
print(f"  {OUT/'human_labels_v2_blank.csv'}  ({len(all_picks)} rows)")
print(f"  {OUT/'claude_labels_v2_for_comparison.csv'}  (answer key, hidden)")
print(f"  {OUT/'user_labels_v2_overlap.csv'}  ({len(overlap_ids)} overlap rows)")
