"""Recompute Stage B headline numbers using:
- user's 30 hand labels for the 30 picked rows
- Claude's labels for the other 60 rows (caveat: unverified under user's interpretation)

Reports per-combo label-faithfulness accuracy under both strict and lenient rules.
"""
from __future__ import annotations
import csv
from pathlib import Path
from collections import defaultdict

OUT = Path("/Users/yangtian/Documents/law/calib/RuleFollower/modernbert_uq/results/audit/stageB")
csv.field_size_limit(2**31 - 1)

human_path = OUT / "human_labels_blank_filled.csv"
claude_path = OUT / "per_branch_labels.csv"

human = {r["row_id"]: r for r in csv.DictReader(human_path.open())}
claude = {r["row_id"]: r for r in csv.DictReader(claude_path.open())}

def L(v):  # normalize labels
    return (v or "").strip().lower()

def is_accept(cwf, sc, lenient=False):
    """Strict: cwf=yes AND sc=yes.   Lenient: cwf=yes AND sc in (yes, partial)."""
    if cwf != "yes": return False
    if lenient: return sc in ("yes", "partial")
    return sc == "yes"

def bin_of(s):
    n = int(s)
    if 1 <= n <= 4: return "1-4"
    if 5 <= n <= 7: return "5-7"
    if 8 <= n <= 9: return "8-9"
    return "?"

# Build a unified label set: prefer human; fall back to Claude
combined = {}
for rid, r in claude.items():
    cwf = L(r["claim_well_formed"])
    sc  = L(r["score_correct"])
    combined[rid] = {
        "row_id": rid, "combo": r["combo"], "score": r["score"],
        "cwf": cwf, "sc": sc, "source": "claude",
    }
for rid, r in human.items():
    cwf = L(r["claim_well_formed"])
    sc  = L(r["score_correct"])
    if cwf == "" and sc == "": continue   # B0037 is fully blank — keep claude's
    # If sc is blank but cwf=yes and ent=yes, infer sc=yes if score 8-9, else use claude
    if sc == "" and cwf == "yes":
        ent = L(r["entailment"])
        score = int(r["score_annotator"])
        if ent == "yes" and 8 <= score <= 9: sc = "yes"
        elif ent == "no"  and 1 <= score <= 3: sc = "yes"
        elif ent == "partial" and 4 <= score <= 7: sc = "yes"
        elif ent != "": sc = "no"
        else: sc = combined[rid]["sc"]   # fallback
    combined[rid] = {
        "row_id": rid, "combo": r["combo"], "score": r["score_annotator"],
        "cwf": cwf if cwf else combined[rid]["cwf"],
        "sc":  sc  if sc  else combined[rid]["sc"],
        "source": "human",
    }

# Per-combo, per-source-of-label accuracy
def acc_by_combo(rows, lenient=False):
    by_combo = defaultdict(list)
    for r in rows:
        by_combo[r["combo"]].append(r)
    out = {}
    for c, rs in by_combo.items():
        n = len(rs)
        a = sum(1 for r in rs if is_accept(r["cwf"], r["sc"], lenient=lenient))
        out[c] = (a, n, 100*a/n if n else 0)
    return out

# View 1 — user-only (30 rows)
user_only = [combined[rid] for rid in human if combined[rid]["source"] == "human"]
# View 2 — claude-only (the 60 not in human)
claude_only = [combined[rid] for rid in claude if rid not in human or combined[rid]["source"] == "claude"]
# View 3 — combined (all 90)
all_rows = list(combined.values())

print("=" * 70)
print("VIEW 1 — User-labeled rows only (30 rows, 10 per combo)")
print("=" * 70)
for label, lenient in [("strict (sc=yes only)", False), ("lenient (sc=yes or partial)", True)]:
    print(f"\n{label}:")
    for c, (a, n, pct) in acc_by_combo(user_only, lenient=lenient).items():
        print(f"  {c:20s} {a}/{n} = {pct:.1f}%")

# Per-bin within user-only
print("\nUser-only — per score bin (strict):")
by_bin_combo = defaultdict(lambda: defaultdict(list))
for r in user_only:
    by_bin_combo[bin_of(r["score"])][r["combo"]].append(r)
for b in ("1-4","5-7","8-9"):
    print(f"  bin {b}:")
    for c, rs in by_bin_combo[b].items():
        a = sum(1 for r in rs if is_accept(r["cwf"], r["sc"]))
        print(f"    {c:20s} {a}/{len(rs)} = {100*a/len(rs):.0f}%")

print("\n" + "=" * 70)
print("VIEW 2 — Claude-labeled rows only (60 rows, 20 per combo)")
print("=" * 70)
print("(Claude's labels — NOT re-judged under user's interpretation)")
for c, (a, n, pct) in acc_by_combo(claude_only, lenient=False).items():
    print(f"  {c:20s} {a}/{n} = {pct:.1f}%")

print("\n" + "=" * 70)
print("VIEW 3 — Combined (30 user + 60 Claude, all 90)")
print("=" * 70)
for label, lenient in [("strict", False), ("lenient", True)]:
    print(f"\n{label}:")
    for c, (a, n, pct) in acc_by_combo(all_rows, lenient=lenient).items():
        print(f"  {c:20s} {a}/{n} = {pct:.1f}%")

# Hypothesis II revisited
print("\n" + "=" * 70)
print("Hypothesis II re-check (LLM > heuristic by ≥ 5 pp ?)")
print("=" * 70)
for view_name, rows in [("user-only", user_only), ("combined", all_rows)]:
    for lenient in (False, True):
        accs = acc_by_combo(rows, lenient=lenient)
        h = accs.get("heuristic", (0,0,0))[2]
        gp = accs.get("llm_gpt5mini", (0,0,0))[2]
        go = accs.get("llm_gptoss", (0,0,0))[2]
        rule = "lenient" if lenient else "strict"
        print(f"\n{view_name} {rule}:")
        print(f"  heuristic     = {h:.1f}%")
        print(f"  llm_gpt5mini  = {gp:.1f}%   gap = {gp-h:+.1f} pp")
        print(f"  llm_gptoss    = {go:.1f}%   gap = {go-h:+.1f} pp")
