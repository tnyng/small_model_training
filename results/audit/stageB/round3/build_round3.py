"""Build round-3 hand-labeling pack for 5 annotators.

Design (max-unique under "each claim labeled 3 times"):
  - 5 overlap claims (already labeled by A1+A2): each gets 1 more label.
    Distributed 2/2/1 across A3/A4/A5 → reaches 3 labels each.
  - 48 new claims (never seen by any human): each labeled by 3 annotators.
    1 of the 48 is labeled by 4 annotators (absorbs the 1-label surplus from
    150 budget − 5 overlap labels = 145 = 47×3 + 4 = 48 unique with 1 quadruple).

Per-annotator total: 30 (10 per score-bin, with up to ±1 deviation).

Outputs in this folder:
  - A1_blank.csv … A5_blank.csv  (each: 30 rows, blind-labeling sheet)
  - assignment_key.csv           (which annotators see which row, hidden)
"""
from __future__ import annotations
import csv, json, random
from pathlib import Path
from collections import defaultdict, Counter

csv.field_size_limit(2**31 - 1)
ROOT = Path("/Users/yangtian/Documents/law/calib/RuleFollower")
DATA = ROOT / "modernbert_uq" / "data"
STAGE_B = ROOT / "modernbert_uq" / "results" / "audit" / "stageB"
OUT = STAGE_B / "round3"
OUT.mkdir(parents=True, exist_ok=True)

# ---- combos to draw from (matches the 3 audited combos) ----
COMBOS = [
    ("heuristic_ann_gpt41mini",      "rulefollower_heuristic_ann_gpt41mini_reg_base__filtered"),
    ("llm_gpt5mini_ann_gpt41mini",   "rulefollower_llm_gpt5mini_ann_gpt41mini_reg_base__filtered"),
    ("llm_gptoss_ann_gpt41mini",     "rulefollower_llm_gptoss_ann_gpt41mini_ge8__filtered"),
]

# ---- excluded row IDs (already labeled by humans in earlier rounds) ----
def load_existing_ids() -> set:
    ids = set()
    for fn in ("human_labels_blank_filled.csv", "sample_label_yu.csv",
               "per_branch_sample.csv"):  # per_branch_sample also defined the v1+v2 pool
        p = STAGE_B / fn
        if p.exists():
            for r in csv.DictReader(p.open()):
                ids.add(r.get("row_id", ""))
    return {x for x in ids if x}

# ---- 5 overlap claims and their bins ----
OVERLAP = {
    # row_id : (combo, score)
    "B0033": ("llm_gpt5mini",  4),  # bin 1-4
    "B0004": ("heuristic",     2),  # bin 1-4
    "B0032": ("llm_gpt5mini",  3),  # bin 1-4
    "B0065": ("llm_gptoss",    3),  # bin 1-4
    "B0015": ("heuristic",     7),  # bin 5-7
}

def bin_of(score):
    s = int(round(float(score)))
    if 1 <= s <= 4: return "1-4"
    if 5 <= s <= 7: return "5-7"
    if 8 <= s <= 9: return "8-9"
    return "?"

# ---- load filtered Layer 1 train rows from each combo, exclude already-labeled ----
def load_pool():
    excluded = load_existing_ids()
    print(f"excluded row_ids (already human-labeled): {len(excluded)}")
    rows = []
    for combo, branch in COMBOS:
        p = DATA / branch / "train.csv"
        if not p.exists():
            print(f"  WARN missing: {p}")
            continue
        for r in csv.DictReader(p.open()):
            if r["id"] in excluded:
                continue
            try:
                score = int(round(float(r.get("accuracy_score", "0"))))
            except Exception:
                continue
            r["_combo"] = combo
            r["_score"] = score
            r["_bin"]   = bin_of(score)
            rows.append(r)
    return rows

# ---- sample 48 new claims, stratified by combo × bin ----
def sample_new(pool, seed=1):
    random.seed(seed)
    by_combo_bin = defaultdict(list)
    for r in pool:
        by_combo_bin[(r["_combo"], r["_bin"])].append(r)
    # Target: 16 per bin × 3 = 48 across 3 combos. Distribute evenly:
    # 6/5/5 per combo per bin (16 total per bin); stretch as needed.
    targets = {
        ("heuristic_ann_gpt41mini",    "1-4"): 6,
        ("heuristic_ann_gpt41mini",    "5-7"): 5,
        ("heuristic_ann_gpt41mini",    "8-9"): 5,
        ("llm_gpt5mini_ann_gpt41mini", "1-4"): 5,
        ("llm_gpt5mini_ann_gpt41mini", "5-7"): 6,
        ("llm_gpt5mini_ann_gpt41mini", "8-9"): 5,
        ("llm_gptoss_ann_gpt41mini",   "1-4"): 5,
        ("llm_gptoss_ann_gpt41mini",   "5-7"): 5,
        ("llm_gptoss_ann_gpt41mini",   "8-9"): 6,
    }
    sampled = []
    for k, n in targets.items():
        pool_k = by_combo_bin.get(k, [])
        if len(pool_k) < n:
            print(f"  WARN {k}: only {len(pool_k)} available, target {n}")
            sampled.extend(pool_k)
        else:
            sampled.extend(random.sample(pool_k, n))
    return sampled

# ---- assign annotators to each new claim ----
def assign_annotators(new_claims, seed=2):
    """
    Each new claim gets 3 annotators (one gets 4 to absorb the +1 surplus).
    Per-annotator new-claim totals: A1=30, A2=30, A3=28, A4=28, A5=29.
    Total = 145 = 47×3 + 1×4.

    Within the per-annotator quota, we also try to get ~10 per bin per annotator.
    We use a greedy assignment over claims sorted by bin difficulty.
    """
    random.seed(seed)
    NEW_TARGETS = {"A1": 30, "A2": 30, "A3": 28, "A4": 28, "A5": 29}
    BIN_TARGET = {
        "A1": {"1-4": 10, "5-7": 10, "8-9": 10},
        "A2": {"1-4": 10, "5-7": 10, "8-9": 10},
        "A3": {"1-4":  8, "5-7": 10, "8-9": 10},   # 2 overlap bin1-4 already
        "A4": {"1-4":  9, "5-7":  9, "8-9": 10},   # 1 overlap bin1-4 + 1 bin5-7
        "A5": {"1-4":  9, "5-7": 10, "8-9": 10},   # 1 overlap bin1-4
    }

    # remaining capacity
    remain_total = dict(NEW_TARGETS)
    remain_bin = {a: dict(BIN_TARGET[a]) for a in NEW_TARGETS}

    # pick which claim gets 4 annotators (the quadruple)
    quad_idx = random.randrange(len(new_claims))

    assignment = []  # list of (claim, [annotators])
    for i, claim in enumerate(new_claims):
        b = claim["_bin"]
        k = 4 if i == quad_idx else 3
        # Score each annotator: prefer those with higher remaining_bin and remaining_total
        candidates = []
        for a in ("A1", "A2", "A3", "A4", "A5"):
            if remain_total[a] <= 0: continue
            score = remain_bin[a].get(b, 0) + 0.5 * remain_total[a] / 30
            # tiebreak with small random
            candidates.append((score + random.random() * 0.01, a))
        candidates.sort(reverse=True)
        chosen = [a for _, a in candidates[:k]]
        for a in chosen:
            remain_total[a] -= 1
            remain_bin[a][b] = max(0, remain_bin[a][b] - 1)
        assignment.append((claim, chosen))

    print(f"\nremaining-after-assignment (should be 0 / minor):")
    for a in ("A1","A2","A3","A4","A5"):
        print(f"  {a}: total={remain_total[a]}  bin={remain_bin[a]}")
    return assignment

# ---- assign overlap claims (5 → 2/2/1) ----
def assign_overlap(seed=3):
    random.seed(seed)
    overlap_ids = list(OVERLAP.keys())
    # A3 takes 2 from bin 1-4 (B0033, B0004)
    # A4 takes 1 from bin 1-4 + 1 from bin 5-7 (B0032, B0015)
    # A5 takes 1 from bin 1-4 (B0065)
    return {
        "B0033": "A3",
        "B0004": "A3",
        "B0032": "A4",
        "B0015": "A4",
        "B0065": "A5",
    }

# ---- build per-annotator CSVs ----
def build_csvs(new_assignment, overlap_assignment):
    # Re-load v2 sample to get the source text + claim text for the 5 overlap rows
    v2 = {r["row_id"]: r for r in csv.DictReader((STAGE_B / "human_labels_v2_blank.csv").open())}

    by_annotator = defaultdict(list)
    # New claims
    for claim, annotators in new_assignment:
        record = {
            "row_id": claim["id"],
            "combo": claim["_combo"],
            "score_annotator": claim["_score"],
            "claim": claim.get("property", ""),
            "source": claim.get("source_text", ""),
            "claim_well_formed": "",
            "entailment": "",
            "score_correct": "",
            "notes": "",
        }
        for a in annotators:
            by_annotator[a].append(record)

    # Overlap claims (only assigned to A3/A4/A5)
    for rid, annotator in overlap_assignment.items():
        v2_row = v2[rid]
        record = {
            "row_id": rid,
            "combo": v2_row["combo"],
            "score_annotator": v2_row["score_annotator"],
            "claim": v2_row["claim"],
            "source": v2_row["source"],
            "claim_well_formed": "",
            "entailment": "",
            "score_correct": "",
            "notes": "",
        }
        by_annotator[annotator].append(record)

    fields = ["row_id","combo","score_annotator","claim","source",
              "claim_well_formed","entailment","score_correct","notes"]
    for a in ("A1","A2","A3","A4","A5"):
        rows = by_annotator[a]
        random.shuffle(rows)  # so overlap rows aren't always at the end
        with (OUT / f"{a}_blank.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows: w.writerow(r)
        bins = Counter(bin_of(r["score_annotator"]) for r in rows)
        print(f"  {a}: {len(rows)} rows  bins={dict(bins)}")

    # Build assignment key (for later merge / kappa)
    keys = []
    for claim, annotators in new_assignment:
        keys.append({
            "row_id": claim["id"],
            "kind": "new",
            "combo": claim["_combo"],
            "score_annotator": claim["_score"],
            "annotators": "|".join(annotators),
            "n_labels_target": str(len(annotators)),
        })
    for rid, annotator in overlap_assignment.items():
        keys.append({
            "row_id": rid,
            "kind": "overlap",
            "combo": OVERLAP[rid][0],
            "score_annotator": OVERLAP[rid][1],
            "annotators": f"A1|A2|{annotator}",   # 2 existing + 1 new
            "n_labels_target": "3",
        })
    with (OUT / "assignment_key.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(keys[0].keys()))
        w.writeheader()
        for r in keys: w.writerow(r)

def main():
    pool = load_pool()
    print(f"eligible new-claim pool size: {len(pool)}")
    new_claims = sample_new(pool)
    print(f"sampled {len(new_claims)} new claims")
    bin_counts = Counter(c["_bin"] for c in new_claims)
    combo_counts = Counter(c["_combo"] for c in new_claims)
    print(f"  bin distribution: {dict(bin_counts)}")
    print(f"  combo distribution: {dict(combo_counts)}")

    new_assignment = assign_annotators(new_claims)
    overlap_assignment = assign_overlap()

    print("\nBuilding per-annotator CSVs:")
    build_csvs(new_assignment, overlap_assignment)
    print(f"\nWrote outputs to: {OUT}")

if __name__ == "__main__":
    main()
