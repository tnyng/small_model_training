"""Merge v2 (second-annotator) labels with Claude's labels and the first
annotator's labels (overlap rows). Compute:

- v2 vs Claude kappa (all 30 v2 rows)
- v2 vs first-annotator kappa (5 overlap rows)
- Combined accuracy per combo using all available human labels

Run after `human_labels_v2_filled.csv` is saved.
"""
from __future__ import annotations
import csv, sys
from pathlib import Path
from collections import defaultdict

OUT = Path("/Users/yangtian/Documents/law/calib/RuleFollower/modernbert_uq/results/audit/stageB")
csv.field_size_limit(2**31 - 1)

v2_path     = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT/"human_labels_v2_filled.csv"
claude_path = OUT/"claude_labels_v2_for_comparison.csv"
user1_overlap = OUT/"user_labels_v2_overlap.csv"
all_claude  = OUT/"per_branch_labels.csv"
user_30     = OUT/"human_labels_blank_filled.csv"

if not v2_path.exists():
    print(f"Could not find {v2_path}. Save your filled CSV as human_labels_v2_filled.csv and re-run.")
    sys.exit(1)

v2     = {r["row_id"]: r for r in csv.DictReader(v2_path.open())}
claude = {r["row_id"]: r for r in csv.DictReader(claude_path.open())}
user1  = {r["row_id"]: r for r in csv.DictReader(user1_overlap.open())}
all_cl = {r["row_id"]: r for r in csv.DictReader(all_claude.open())}
user1_full = {r["row_id"]: r for r in csv.DictReader(user_30.open())}

def L(v): return ((v or "").strip().lower())
def cell(d, k):
    v = L(d.get(k,""))
    return None if v in ("","n/a","na") else v

def cohens_kappa(a, b):
    n = len(a)
    if n == 0: return None, 0
    cats = sorted(set(a) | set(b))
    if len(cats) < 2: return 1.0, n
    po = sum(1 for x,y in zip(a,b) if x==y) / n
    pe = sum((a.count(c)/n) * (b.count(c)/n) for c in cats)
    if pe == 1: return 1.0, n
    return (po-pe)/(1-pe), n

# v2 vs Claude (all 30)
print("=== v2 (second annotator) vs Claude — all 30 v2 rows ===")
fields = [("claim_well_formed","claim_well_formed_claude"),
          ("entailment","entailment_claude"),
          ("score_correct","score_correct_claude")]
for hf, cf in fields:
    av, ac = [], []
    for rid, r in v2.items():
        h = cell(r, hf); c = cell(claude[rid], cf)
        if h is None or c is None: continue
        av.append(h); ac.append(c)
    k, n = cohens_kappa(av, ac)
    agree = sum(1 for x,y in zip(av,ac) if x==y)
    print(f"  {hf}: kappa={k:.3f}  raw={agree}/{n} = {100*agree/n:.0f}%")

# v2 vs user1 (overlap 5 rows)
print("\n=== v2 vs first annotator — 5 overlap rows ===")
overlap_ids = [rid for rid in v2 if rid in user1]
print(f"Overlap rows: {overlap_ids}")
for hf, _ in fields:
    av, au = [], []
    for rid in overlap_ids:
        h = cell(v2[rid], hf); u = cell(user1[rid], hf+"_user")
        if h is None or u is None: continue
        av.append(h); au.append(u)
    k, n = cohens_kappa(av, au)
    agree = sum(1 for x,y in zip(av,au) if x==y) if n else 0
    print(f"  {hf}: kappa={k:.3f}  raw={agree}/{n} = {100*agree/n if n else 0:.0f}%")

# Combined accuracy: user-30 + v2-30 + claude on the rest (~30)
print("\n=== Combined accuracy per combo (user-30 + v2-30 + claude on rest) ===")
def is_accept(cwf, sc, lenient=False):
    if cwf != "yes": return False
    return sc in ("yes","partial") if lenient else sc == "yes"

def label_for(rid):
    """Best label: prefer human (user1 first-annotator wins ties), then v2, then claude."""
    if rid in user1_full and L(user1_full[rid]["claim_well_formed"]):
        r = user1_full[rid]
        return r["combo"], r["score_annotator"], L(r["claim_well_formed"]), L(r["score_correct"])
    if rid in v2 and L(v2[rid]["claim_well_formed"]):
        r = v2[rid]
        return r["combo"], r["score_annotator"], L(r["claim_well_formed"]), L(r["score_correct"])
    r = all_cl[rid]
    return r["combo"], r["score"], L(r["claim_well_formed"]), L(r["score_correct"])

by_combo = defaultdict(list)
for rid in all_cl:
    combo, score, cwf, sc = label_for(rid)
    by_combo[combo].append((cwf, sc))

for lenient in (False, True):
    print(f"\n  {'lenient' if lenient else 'strict'}:")
    for c, rs in by_combo.items():
        a = sum(1 for cwf, sc in rs if is_accept(cwf, sc, lenient))
        n = len(rs)
        print(f"    {c:20s} {a}/{n} = {100*a/n:.1f}%")

# Hypothesis II re-check
print("\n=== Hypothesis II re-check ===")
for lenient in (False, True):
    label = "lenient" if lenient else "strict"
    accs = {c: sum(1 for cwf, sc in rs if is_accept(cwf, sc, lenient))/len(rs)*100
            for c, rs in by_combo.items()}
    h = accs.get("heuristic", 0)
    print(f"  {label}: heuristic={h:.1f}%   gpt-5-mini gap={accs.get('llm_gpt5mini',0)-h:+.1f}pp   gpt-oss gap={accs.get('llm_gptoss',0)-h:+.1f}pp")

# Save full disagreement CSV for v2 vs Claude
with (OUT/"v2_vs_claude_disagreements.csv").open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["row_id","combo","score","field","v2","claude","note_claude"])
    for rid, r in v2.items():
        for hf, cf in fields:
            h = cell(r, hf); c = cell(claude[rid], cf)
            if h is None or c is None or h == c: continue
            w.writerow([rid, r["combo"], r["score_annotator"], hf, h, c, claude[rid].get("note_claude","")])
print(f"\nv2-vs-Claude disagreements written to {OUT/'v2_vs_claude_disagreements.csv'}")
