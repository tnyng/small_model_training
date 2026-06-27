"""Stage D — branch triage. Apply Stage 0 §1b thresholds and Stage C parser
decision to all 34 modernbert experiment branches. Output branch_triage.csv.
"""
from __future__ import annotations
import csv, json
from pathlib import Path
from collections import OrderedDict

ROOT = Path("/Users/yangtian/Documents/law/calib/RuleFollower")
A_DIR  = ROOT / "modernbert_uq" / "results" / "audit" / "stageA" / "per_experiment_branch"
A5_SUM = ROOT / "modernbert_uq" / "results" / "audit" / "stageA5" / "apply_summary.json"
OUT_DIR = ROOT / "modernbert_uq" / "results" / "audit" / "stageD"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- inputs ----------
a5 = json.loads(A5_SUM.read_text())["branches"]

stageB_lf = {
    "heuristic_ann_gpt41mini":      0.50,   # combined-lenient view (most data)
    "llm_gpt5mini_ann_gpt41mini":   0.733,
    "llm_gptoss_ann_gpt41mini":     0.60,
}

# Map combo string to parser_family / annotator / parser_role (from Stage C)
COMBO_INFO = {
    "heuristic_ann_gpt41mini":     ("heuristic",     "gpt41mini", "robustness", "audited"),
    "heuristic_ann_gpt5mini":      ("heuristic",     "gpt5mini",  "robustness", "annotator-side not audited"),
    "heuristic_ann_gptoss":        ("heuristic",     "gptoss",    "robustness", "annotator-side not audited"),
    "llm_gpt5mini_ann_gpt41mini":  ("llm_gpt5mini",  "gpt41mini", "main",       "audited"),
    "llm_gpt5mini_ann_gpt5mini":   ("llm_gpt5mini",  "gpt5mini",  "main",       "annotator-side not audited"),
    "llm_gptoss_ann_gpt41mini":    ("llm_gptoss",    "gpt41mini", "secondary",  "audited"),
    "llm_gptoss_ann_gptoss":       ("llm_gptoss",    "gptoss",    "secondary",  "annotator-side not audited"),
}

def parse_branch(name: str):
    # name like "rulefollower_heuristic_ann_gpt41mini_reg_base"
    rest = name.replace("rulefollower_", "")
    for combo in COMBO_INFO:
        if rest.startswith(combo + "_"):
            variant = rest[len(combo)+1:]
            return combo, variant
    return None, None

# Stage 0 thresholds (locked)
TRAIN_DROP_LT   = 1000
TRAIN_CAVEAT_LT = 5000
TID_LEAK_DROP   = 0.01
TID_LEAK_CAVEAT = 0.001
DUP_DROP        = 0.05
LF_DROP         = 0.60
LF_CAVEAT       = 0.75
MAX_REP_CAVEAT  = 5

def triage(row) -> tuple[str, list[str]]:
    reasons = []
    drop_reasons = []
    caveat_reasons = []
    # train rows (post-filter)
    if row["train_post_filter"] is not None:
        if row["train_post_filter"] < TRAIN_DROP_LT:
            drop_reasons.append(f"post-filter train < {TRAIN_DROP_LT} (={row['train_post_filter']})")
        elif row["train_post_filter"] < TRAIN_CAVEAT_LT:
            caveat_reasons.append(f"post-filter train < {TRAIN_CAVEAT_LT} (={row['train_post_filter']})")
    # text_id leakage
    if row["tid_leak_pct"] > TID_LEAK_DROP:
        drop_reasons.append(f"text_id leakage > {100*TID_LEAK_DROP:.0f}% (={100*row['tid_leak_pct']:.2f}%)")
    elif row["tid_leak_pct"] > TID_LEAK_CAVEAT:
        caveat_reasons.append(f"text_id leakage 0.1–1% (={100*row['tid_leak_pct']:.2f}%)")
    # property dup rate (post-rebalancing)
    if row["prop_dup_rate"] > DUP_DROP:
        # Only flag for rebalanced slices since reg_base et al naturally have ~3-5%
        if row["max_id_rep"] > 1:
            drop_reasons.append(f"post-rebal dup rate > {100*DUP_DROP:.0f}% with id-repetition ({row['max_id_rep']}x)")
    # max id repetition
    if row["max_id_rep"] > MAX_REP_CAVEAT:
        caveat_reasons.append(f"single source claim duplicated {row['max_id_rep']}x in train")
    # label-faithfulness (only if available)
    if row["lf_accuracy"] is not None:
        if row["lf_accuracy"] < LF_DROP:
            drop_reasons.append(f"label-faithfulness accuracy < {100*LF_DROP:.0f}% (={100*row['lf_accuracy']:.0f}%)")
        elif row["lf_accuracy"] < LF_CAVEAT:
            caveat_reasons.append(f"label-faithfulness accuracy {100*LF_DROP:.0f}–{100*LF_CAVEAT:.0f}% (={100*row['lf_accuracy']:.0f}%)")
    # parser-role override
    if row["parser_role"] == "robustness":
        caveat_reasons.append("parser=heuristic; Stage C dropped as primary, kept for robustness/appendix only")
    decision = "drop" if drop_reasons else ("caveat" if caveat_reasons else "keep")
    return decision, drop_reasons + caveat_reasons

# ---------- build rows ----------
out_rows = []
for jf in sorted(A_DIR.glob("*.json")):
    d = json.loads(jf.read_text())
    name = d["branch"]
    combo, variant = parse_branch(name)
    if not combo: continue
    parser_family, annotator, parser_role, audit_status = COMBO_INFO[combo]
    a5b = a5.get(name, {})
    row = {
        "branch": name,
        "parser_family": parser_family,
        "annotator": annotator,
        "variant": variant or "",
        "parser_role": parser_role,
        "audit_status": audit_status,
        "train_pre_filter": d["rebalancing_sanity"]["train_rows"],
        "train_post_filter": a5b.get("train_kept"),
        "post_filter_drop_rate": round(a5b.get("drop_rate", 0), 3) if a5b else None,
        "max_id_rep": d["rebalancing_sanity"]["max_id_repetition"],
        "tid_leak_pct": d["split_sanity"]["text_id_leak_pct_of_train"],
        "text_leak_pct": d["split_sanity"]["text_leak_pct_of_train"],
        "prop_dup_rate": d["within_train_duplication"]["property_dup_rate"],
        "lf_accuracy": stageB_lf.get(combo),
    }
    decision, reasons = triage(row)
    row["decision"] = decision
    row["reasons"] = " | ".join(reasons) if reasons else "all thresholds passed"
    out_rows.append(row)

# Write CSV
csv_fields = ["branch","parser_family","annotator","variant","parser_role","audit_status",
              "train_pre_filter","train_post_filter","post_filter_drop_rate",
              "max_id_rep","tid_leak_pct","prop_dup_rate","lf_accuracy",
              "decision","reasons"]
with (OUT_DIR/"branch_triage.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=csv_fields)
    w.writeheader()
    for r in out_rows: w.writerow({k: r.get(k,"") for k in csv_fields})

# Print summary table
print(f"Triage summary ({len(out_rows)} branches)")
from collections import Counter
dec = Counter(r["decision"] for r in out_rows)
print(f"  keep:   {dec.get('keep',0)}")
print(f"  caveat: {dec.get('caveat',0)}")
print(f"  drop:   {dec.get('drop',0)}")

print("\n=== Per-branch table ===")
print(f"{'branch':62s} {'role':12s} {'tr_in':>6s} {'tr_post':>7s} {'maxrep':>6s} {'lf':>5s} {'decision':>8s}")
for r in out_rows:
    lf = f"{100*r['lf_accuracy']:.0f}%" if r["lf_accuracy"] is not None else "  -"
    tr_post = str(r["train_post_filter"]) if r["train_post_filter"] is not None else "?"
    print(f"{r['branch']:62s} {r['parser_role']:12s} {r['train_pre_filter']:>6d} {tr_post:>7s} {r['max_id_rep']:>6d} {lf:>5s} {r['decision']:>8s}")
print(f"\nTriage CSV: {OUT_DIR/'branch_triage.csv'}")
