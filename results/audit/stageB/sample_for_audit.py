"""Stage B sampling — draws stratified claims for Tier 2 LLM-auditor judgment.

For each first-pass branch:
- 30 claims stratified: 10 from score 1-4, 5 from 5-7, 10 from 8-9, 5 from Stage-A flag pool.

Plus paired-article comparison: 5 text_ids present in all 3 main combos; pull all
claims each parser produced for those articles.

Operates on the FILTERED Layer 2 corpus (apply the Stage A.5 filter in-flight, since
Layer 2 doesn't have a __filtered sibling).
"""
from __future__ import annotations
import csv, json, random, re, sys
from pathlib import Path
from collections import defaultdict, Counter

sys.path.insert(0, str(Path(__file__).parent.parent / "stageA5"))
from filter_claims import evaluate_claim, lemma_proxy_coverage

csv.field_size_limit(2**31 - 1)

ROOT = Path("/Users/yangtian/Documents/law/calib/RuleFollower")
EXP = ROOT / "experiments"
OUT = ROOT / "modernbert_uq" / "results" / "audit" / "stageB"

import hashlib
def text_id_hash(s):
    return hashlib.md5((s or "").strip().encode("utf-8")).hexdigest()[:16]

ANS_RE = re.compile(r"\[?Answer\]?:?\s*([0-9]+)", re.IGNORECASE)
def parse_score(out):
    m = ANS_RE.search(out or "")
    try: return int(m.group(1)) if m else None
    except: return None

COMBOS = [
    ("heuristic_ann_gpt41mini",       "annotation_gpt4.1mini_heuristic"),
    ("llm_gpt5mini_ann_gpt41mini",    "annotation_gpt4.1mini_gpt-5-mini"),
    ("llm_gptoss_ann_gpt41mini",      "annotation_gpt4.1mini_gpt-oss"),
]

def load_filtered(folder: str):
    """Load combo's Layer 2 rows, apply Stage A.5 filter, keep only kept rows.
    Tag each kept row with parsed score and text_id hash."""
    rows = []
    base = EXP / folder / "accuracy_outputs"
    for f in sorted(base.glob("*.csv")):
        ds = f.stem.split("_score_")[0]
        for r in csv.DictReader(f.open()):
            r["_dataset"] = ds
            ev = evaluate_claim(r.get("property",""), r.get("text",""))
            r["_dropped"] = ev["drop"]
            r["_review"] = ev["review"]
            r["_coverage"] = round(ev["coverage"], 3)
            r["_score"] = parse_score(r.get("output",""))
            r["_tid"] = text_id_hash(r.get("text",""))
            if not ev["drop"]:
                rows.append(r)
    return rows

def sample_branch(combo_name: str, rows: list, seed=42):
    random.seed(hash(combo_name) ^ seed)
    by_bin = {"1-4": [], "5-7": [], "8-9": []}
    flag_pool = []
    for r in rows:
        s = r["_score"]
        if s is None: continue
        if 1 <= s <= 4: by_bin["1-4"].append(r)
        elif 5 <= s <= 7: by_bin["5-7"].append(r)
        elif 8 <= s <= 9: by_bin["8-9"].append(r)
        # flag pool: review-flagged or low coverage
        if r["_review"] or (0 < r["_coverage"] <= 0.20):
            flag_pool.append(r)
    samples = []
    targets = {"1-4": 10, "5-7": 5, "8-9": 10}
    for b, k in targets.items():
        pool = by_bin[b]
        samples.extend(random.sample(pool, min(k, len(pool))))
    # 5 from flag pool
    if flag_pool:
        samples.extend(random.sample(flag_pool, min(5, len(flag_pool))))
    return samples

def main():
    combo_data = {name: load_filtered(folder) for name, folder in COMBOS}
    print("Filtered row counts (kept):")
    for k,v in combo_data.items(): print(f"  {k}: {len(v)}")

    # Per-branch samples
    all_per_branch = {}
    for name, rows in combo_data.items():
        s = sample_branch(name, rows)
        all_per_branch[name] = s
        print(f"  sampled {len(s)} from {name}")

    # Paired article comparison: tids present in all three combos
    tid_sets = {name: {r["_tid"] for r in rows} for name, rows in combo_data.items()}
    common_tids = sorted(set.intersection(*tid_sets.values()))
    print(f"  shared text_ids across all 3 combos: {len(common_tids)}")
    random.seed(0)
    paired_tids = random.sample(common_tids, min(5, len(common_tids)))
    paired_rows = []
    for tid in paired_tids:
        for name, rows in combo_data.items():
            for r in rows:
                if r["_tid"] == tid:
                    paired_rows.append({**r, "_combo": name})
    print(f"  paired-article rows total: {len(paired_rows)}")

    # Write samples
    OUT.mkdir(parents=True, exist_ok=True)
    fieldnames = ["row_id","combo","dataset","text_id","score","coverage","review",
                  "property","source","reasoning_content"]
    out_rows = []
    rid = 0
    for name, samp in all_per_branch.items():
        for r in samp:
            rid += 1
            out_rows.append({
                "row_id": f"B{rid:04d}",
                "combo": name,
                "dataset": r["_dataset"],
                "text_id": r["_tid"],
                "score": r["_score"],
                "coverage": r["_coverage"],
                "review": "1" if r["_review"] else "",
                "property": r.get("property",""),
                "source": (r.get("text","") or "")[:1500],
                "reasoning_content": (r.get("reasoning_content","") or "")[:800],
            })
    with (OUT/"per_branch_sample.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader()
        for r in out_rows: w.writerow(r)
    print(f"\nWrote per_branch_sample.csv: {len(out_rows)} rows")

    out_paired = []
    rid = 0
    for r in paired_rows:
        rid += 1
        out_paired.append({
            "row_id": f"P{rid:04d}",
            "combo": r["_combo"],
            "dataset": r["_dataset"],
            "text_id": r["_tid"],
            "score": r["_score"],
            "coverage": r["_coverage"],
            "review": "1" if r["_review"] else "",
            "property": r.get("property",""),
            "source": (r.get("text","") or "")[:1500],
            "reasoning_content": (r.get("reasoning_content","") or "")[:800],
        })
    with (OUT/"paired_article_sample.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader()
        for r in out_paired: w.writerow(r)
    print(f"Wrote paired_article_sample.csv: {len(out_paired)} rows")

if __name__ == "__main__":
    main()
