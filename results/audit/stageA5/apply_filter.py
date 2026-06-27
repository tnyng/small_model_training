"""Apply the locked Stage A.5 filter to:
- all 7 raw (parser, annotator) combos in experiments/ (writes per-combo filter logs)
- all 34 modernbert experiment branches in modernbert_uq/data/ (writes filtered siblings)
"""
from __future__ import annotations
import csv, json, os, sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from filter_claims import evaluate_claim

csv.field_size_limit(2**31 - 1)

ROOT = Path("/Users/yangtian/Documents/law/calib/RuleFollower")
EXP = ROOT / "experiments"
MB = ROOT / "modernbert_uq" / "data"
OUT = ROOT / "modernbert_uq" / "results" / "audit" / "stageA5"
RAW_LOG = OUT / "raw_combo_logs"
RAW_LOG.mkdir(parents=True, exist_ok=True)

RAW_COMBOS = [
    ("heuristic_ann_gpt41mini",       "annotation_gpt4.1mini_heuristic"),
    ("heuristic_ann_gpt5mini",        "annotation_gpt5mini_heuristic"),
    ("heuristic_ann_gptoss",          "annotation_gpt-oss_heuristic"),
    ("llm_gpt5mini_ann_gpt41mini",    "annotation_gpt4.1mini_gpt-5-mini"),
    ("llm_gpt5mini_ann_gpt5mini",     "annotation_gpt5mini"),
    ("llm_gptoss_ann_gpt41mini",      "annotation_gpt4.1mini_gpt-oss"),
    ("llm_gptoss_ann_gptoss",         "annotation_gpt-oss"),
]

import re
ANS_RE = re.compile(r"\[?Answer\]?:?\s*([0-9]+)", re.IGNORECASE)

def parse_score(out):
    m = ANS_RE.search(out or "")
    try: return int(m.group(1)) if m else None
    except: return None

def filter_raw_combo(combo_name, ann_folder):
    folder = EXP / ann_folder / "accuracy_outputs"
    if not folder.exists():
        return None
    rule_drop = Counter()
    rule_review = Counter()
    bin_pre = Counter(); bin_post = Counter()
    n_total = 0; n_drop = 0; n_review = 0
    by_dataset = defaultdict(lambda: {"total":0, "drop":0, "review":0})
    for f in sorted(folder.glob("*.csv")):
        ds = f.stem.split("_score_")[0]
        for r in csv.DictReader(f.open()):
            n_total += 1
            by_dataset[ds]["total"] += 1
            score = parse_score(r.get("output",""))
            if score:
                if score <= 4: bin_pre["1-4"] += 1
                elif score <= 7: bin_pre["5-7"] += 1
                else: bin_pre["8-9"] += 1
            ev = evaluate_claim(r.get("property",""), r.get("text",""))
            if ev["drop"]:
                n_drop += 1
                by_dataset[ds]["drop"] += 1
                for ru in ev["rules_fired"]: rule_drop[ru] += 1
            else:
                if score:
                    if score <= 4: bin_post["1-4"] += 1
                    elif score <= 7: bin_post["5-7"] += 1
                    else: bin_post["8-9"] += 1
                if ev["review"]:
                    n_review += 1
                    by_dataset[ds]["review"] += 1
                    for ru in ev["rules_fired"]: rule_review[ru] += 1
    pre_total = sum(bin_pre.values()) or 1
    post_total = sum(bin_post.values()) or 1
    return {
        "combo": combo_name,
        "rows_total": n_total,
        "rows_dropped": n_drop,
        "rows_review": n_review,
        "rows_kept": n_total - n_drop,
        "drop_rate": n_drop / n_total if n_total else 0,
        "rule_drop_counts": dict(rule_drop),
        "rule_review_counts": dict(rule_review),
        "bin_distribution_pre": {k: bin_pre[k]/pre_total for k in ("1-4","5-7","8-9")},
        "bin_distribution_post": {k: bin_post[k]/post_total for k in ("1-4","5-7","8-9")},
        "bin_distribution_pre_counts": dict(bin_pre),
        "bin_distribution_post_counts": dict(bin_post),
        "by_source_dataset": dict(by_dataset),
    }

def filter_experiment_branch(branch_dir: Path):
    name = branch_dir.name
    out_dir = MB / f"{name}__filtered"
    out_dir.mkdir(exist_ok=True)
    stats = {"branch": name, "splits": {}, "rule_drop_counts": Counter(),
             "bin_pre": Counter(), "bin_post": Counter()}
    for split in ("train", "validation", "test"):
        in_p = branch_dir / f"{split}.csv"
        out_p = out_dir / f"{split}.csv"
        if not in_p.exists(): continue
        rdr = csv.DictReader(in_p.open())
        fieldnames = rdr.fieldnames + ["__filter_review"] if rdr.fieldnames else None
        rows_in = list(rdr)
        n_in = len(rows_in)
        n_drop = 0
        kept = []
        for r in rows_in:
            # bin pre
            try:
                s = int(round(float(r.get("accuracy_score","0"))))
                if 1<=s<=4: stats["bin_pre"]["1-4"] += 1
                elif 5<=s<=7: stats["bin_pre"]["5-7"] += 1
                elif 8<=s<=9: stats["bin_pre"]["8-9"] += 1
            except:
                pass
            ev = evaluate_claim(r.get("property",""), r.get("source_text",""))
            if ev["drop"]:
                n_drop += 1
                for ru in ev["rules_fired"]: stats["rule_drop_counts"][ru] += 1
            else:
                r["__filter_review"] = "1" if ev["review"] else ""
                kept.append(r)
                try:
                    s = int(round(float(r.get("accuracy_score","0"))))
                    if 1<=s<=4: stats["bin_post"]["1-4"] += 1
                    elif 5<=s<=7: stats["bin_post"]["5-7"] += 1
                    elif 8<=s<=9: stats["bin_post"]["8-9"] += 1
                except:
                    pass
        # write
        if kept:
            with out_p.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames or list(kept[0].keys()))
                w.writeheader()
                for r in kept: w.writerow(r)
        else:
            # write header-only to mark intentional empty
            with out_p.open("w", newline="") as f:
                if rows_in:
                    w = csv.DictWriter(f, fieldnames=fieldnames or list(rows_in[0].keys()))
                    w.writeheader()
        stats["splits"][split] = {"in": n_in, "dropped": n_drop, "kept": n_in - n_drop}
    stats["rule_drop_counts"] = dict(stats["rule_drop_counts"])
    pre_total = sum(stats["bin_pre"].values()) or 1
    post_total = sum(stats["bin_post"].values()) or 1
    stats["bin_pre_pct"] = {k: stats["bin_pre"][k]/pre_total for k in ("1-4","5-7","8-9")}
    stats["bin_post_pct"] = {k: stats["bin_post"][k]/post_total for k in ("1-4","5-7","8-9")}
    stats["bin_pre"] = dict(stats["bin_pre"])
    stats["bin_post"] = dict(stats["bin_post"])
    return stats

def main():
    print("=== Applying filter to 7 raw combos ===")
    raw_results = {}
    for combo_name, ann_folder in RAW_COMBOS:
        print(f"  [raw] {combo_name} ...", flush=True)
        r = filter_raw_combo(combo_name, ann_folder)
        if r:
            (RAW_LOG / f"{combo_name}.json").write_text(json.dumps(r, indent=2))
            raw_results[combo_name] = r
    # Print compact summary
    print("\n=== RAW COMBO FILTER SUMMARY ===")
    print(f"{'combo':40s} {'rows_in':>8s} {'dropped':>8s} {'drop%':>7s} {'review':>7s} {'8-9 pre':>9s} {'8-9 post':>9s}")
    for k,v in raw_results.items():
        post = v.get("bin_distribution_post", {}).get("8-9", 0)
        pre  = v.get("bin_distribution_pre",  {}).get("8-9", 0)
        print(f"{k:40s} {v['rows_total']:>8d} {v['rows_dropped']:>8d} {100*v['drop_rate']:>6.1f}% {v['rows_review']:>7d} {100*pre:>8.1f}% {100*post:>8.1f}%")

    print("\n=== Applying filter to 34 modernbert branches ===")
    branch_results = {}
    branch_dirs = sorted([d for d in MB.iterdir()
                          if d.is_dir() and not d.name.endswith("__filtered")
                          and not d.name.startswith("misinfo_")])
    for d in branch_dirs:
        if not (d/"train.csv").exists():
            continue
        print(f"  [branch] {d.name} ...", flush=True)
        s = filter_experiment_branch(d)
        (OUT / f"branch_filter_log_{d.name}.json").write_text(json.dumps(s, indent=2))
        branch_results[d.name] = s

    # Top-level summary
    summary = {
        "raw_combos": {k: {
            "rows_in": v["rows_total"],
            "rows_dropped": v["rows_dropped"],
            "drop_rate": v["drop_rate"],
            "bin_pre": v["bin_distribution_pre"],
            "bin_post": v["bin_distribution_post"],
            "top_rules": Counter(v["rule_drop_counts"]).most_common(5),
        } for k,v in raw_results.items()},
        "branches": {k: {
            "train_in": v["splits"]["train"]["in"],
            "train_kept": v["splits"]["train"]["kept"],
            "drop_rate": (v["splits"]["train"]["dropped"]/v["splits"]["train"]["in"]) if v["splits"]["train"]["in"] else 0,
            "bin_pre_pct": v["bin_pre_pct"],
            "bin_post_pct": v["bin_post_pct"],
        } for k,v in branch_results.items()},
    }
    (OUT / "apply_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nDone. Filtered branches written under modernbert_uq/data/<name>__filtered/")
    print(f"Per-branch logs and summary at {OUT}")

if __name__ == "__main__":
    main()
