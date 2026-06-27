"""Stage A.5 — claim-level filter.

Defines per-claim rules. Apply via `evaluate_claim(prop, source)` → dict of:
  - drop: bool (auto-drop)
  - review: bool (flag for human/auditor review, do NOT auto-drop)
  - rules_fired: list[str]
  - coverage: float

Run as a script to validate against:
  experiments/analysis/noisy_heuristic_claims_three_models_sample_20.csv
plus a stratified random sample drawn from heuristic_ann_gpt41mini Layer 2.

Outputs:
  results/audit/stageA5/filter_validation.json
  results/audit/stageA5/random_drop_review.csv  (for human spot-check, precision input)
"""
from __future__ import annotations
import csv, json, os, re, random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/Users/yangtian/Documents/law/calib/RuleFollower")
EXP = ROOT / "experiments"
OUT = ROOT / "modernbert_uq" / "results" / "audit" / "stageA5"
OUT.mkdir(parents=True, exist_ok=True)
csv.field_size_limit(2**31 - 1)

STOPWORDS = set("""
a an the of in on at to for from with by and or but if then so as is are was were be been being
this that these those it its it's they them their there here will would could should may might can
do does did done not no nor of out over under up down off into out onto about above below before
after between among
""".split())
META_TOKENS = set("""
text article phrase mention mentions mentioning relevant irrelevant discusses discuss discussing
contains contain containing includes include including references reference referencing says said
note notes general generally specific specifically using statement content topic author authors
speaker writer
""".split())

def alpha_words(s):
    return re.findall(r"[A-Za-z][A-Za-z'\-]{2,}", s or "")

def lemma_proxy_coverage(prop, source):
    if not prop or not source:
        return 1.0
    toks = [t.lower() for t in alpha_words(prop)
            if t.lower() not in STOPWORDS and t.lower() not in META_TOKENS]
    if not toks: return 1.0
    src = source.lower()
    return sum(1 for t in toks if t in src) / len(toks)

# --- Rules ---
# Each rule: (name, regex/predicate, action)  where action ∈ {"drop","review"}
# Predicates take (prop_lower, source) and return bool.

def _starts(p, s): return p.startswith(s)
def _matches(p, pattern): return bool(re.match(pattern, p))

RULES = [
    # Templated noise (auto-drop)
    # "the text uses the phrase X" — drop only if X isn't in source. A faithful "uses the
    # phrase" claim is a valid atomic positive for the RFChecker; only the unfaithful ones are noise.
    ("template_uses_phrase_no_cov", lambda p,s,c: _matches(p, r"^the text uses the phrase") and c == 0, "drop"),
    ("template_vacuous_tag",      lambda p,s,c: _matches(p, r"^the text is (a tweet|relevant|irrelevant)\.?\s*$"), "drop"),
    ("template_stutter_the_text", lambda p,s,c: _matches(p, r"^the text the\b"),                              "drop"),
    ("template_negative_no_cov",  lambda p,s,c: _matches(p, r"^the text does not (mention|reference|discuss|contain|include)") and c == 0, "drop"),
    # Awkward starts (drop)
    ("awkward_the_text_is_x",     lambda p,s,c: _matches(p, r"^the text is (possible|likely|unlikely|maybe)"), "drop"),
    ("awkward_the_text_wh",       lambda p,s,c: _matches(p, r"^the text (what|how|when|where|why|whose|which)\b"),  "drop"),
    ("awkward_the_text_that",     lambda p,s,c: _matches(p, r"^the text (that|this) (would|could) (imply|mean|suggest)"), "drop"),
    # Reasoning leakage / scratchpad (drop)
    ("reasoning_leak_start",      lambda p,s,c: _matches(p, r"^(?:the text )?(i should|let me|looking at|without (?:knowing|more)|i need to|i would|i'm not sure)\b"), "drop"),
    # Hypothetical / question (drop)
    ("hypothetical_what_if",      lambda p,s,c: " what if " in p or _matches(p, r"^the text what if\b"),       "drop"),
    ("ends_with_question",        lambda p,s,c: p.rstrip(".").endswith("?"),                                  "drop"),
    ("rhetorical_right_q",        lambda p,s,c: bool(re.search(r"\bright\?+\.?$", p)),                        "drop"),
    # Length/shape (drop)
    ("too_short_lt5_tokens",      lambda p,s,c: len((p or "").split()) < 5,                                   "drop"),
    ("too_long_gt50_tokens",      lambda p,s,c: len((p or "").split()) > 50,                                  "drop"),
    # Coverage (drop / review)
    ("coverage_zero",             lambda p,s,c: c == 0 and len(alpha_words(p)) >= 4,                          "drop"),
    ("coverage_low",              lambda p,s,c: 0 < c <= 0.10 and len(alpha_words(p)) >= 4,                   "review"),
]

def evaluate_claim(prop, source):
    p = (prop or "").strip().lower()
    s = source or ""
    cov = lemma_proxy_coverage(prop, source)
    fired = []
    drop = False
    review = False
    for name, pred, action in RULES:
        try:
            ok = pred(p, s, cov)
        except Exception:
            ok = False
        if ok:
            fired.append(name)
            if action == "drop": drop = True
            elif action == "review": review = True
    return {"drop": drop, "review": review, "rules_fired": fired, "coverage": cov}

# --- Validation ---
def validate_against_noise_sample():
    p = EXP / "analysis" / "noisy_heuristic_claims_three_models_sample_20.csv"
    rows = list(csv.DictReader(p.open()))
    caught = 0
    detail = []
    for r in rows:
        out = evaluate_claim(r.get("property",""), r.get("text",""))
        # All 20 are labeled noisy (noise_score ≥ 2). caught = drop OR review.
        # Strict recall = drop only; lenient recall = drop or review.
        is_caught_drop = out["drop"]
        is_caught_any = out["drop"] or out["review"]
        caught += int(is_caught_drop)
        detail.append({
            "sample_id": r["sample_id"],
            "noise_score": r["noise_score"],
            "noise_patterns": r["noise_patterns"],
            "property": r["property"][:160],
            "drop": is_caught_drop,
            "review": out["review"],
            "rules_fired": out["rules_fired"],
            "coverage": round(out["coverage"], 3),
        })
    n = len(rows)
    return {
        "noise_sample_size": n,
        "n_caught_drop": caught,
        "recall_drop_only": caught / n if n else 0,
        "n_caught_any": sum(1 for d in detail if d["drop"] or d["review"]),
        "recall_any": sum(1 for d in detail if d["drop"] or d["review"]) / n if n else 0,
        "detail": detail,
    }

def stratified_random_sample(combo_folder: str, k_per_bin=50, seed=0):
    """Sample claims from one Layer-2 raw combo for precision spot-check."""
    random.seed(seed)
    folder = EXP / combo_folder / "accuracy_outputs"
    rows = []
    for f in sorted(folder.glob("*.csv")):
        rs = list(csv.DictReader(f.open()))
        for r in rs: r["_dataset"] = f.stem.split("_score_")[0]
        rows.extend(rs)
    # parse score
    ans_re = re.compile(r"\[?Answer\]?:?\s*([0-9]+)", re.IGNORECASE)
    by_bin = {"1-4": [], "5-7": [], "8-9": []}
    for r in rows:
        m = ans_re.search(r.get("output") or "")
        try: s = int(m.group(1)) if m else None
        except: s = None
        r["_score"] = s
        if s is None: continue
        if 1 <= s <= 4: by_bin["1-4"].append(r)
        elif 5 <= s <= 7: by_bin["5-7"].append(r)
        elif 8 <= s <= 9: by_bin["8-9"].append(r)
    sample = []
    for b, pool in by_bin.items():
        sample.extend(random.sample(pool, min(k_per_bin, len(pool))))
    return sample

def main():
    val = validate_against_noise_sample()
    print(f"=== Validation on noisy-claims-20 sample ===")
    print(f"  recall (drop only):       {val['n_caught_drop']}/{val['noise_sample_size']} = {val['recall_drop_only']:.2f}")
    print(f"  recall (drop or review):  {val['n_caught_any']}/{val['noise_sample_size']} = {val['recall_any']:.2f}")
    print()
    # which rules fired most across the noise sample
    rule_hits = Counter()
    for d in val["detail"]:
        for r in d["rules_fired"]: rule_hits[r] += 1
    print("Rule firings on noise sample (top):")
    for r,c in rule_hits.most_common():
        print(f"  {r}: {c}")
    print()
    # claims NOT caught
    missed = [d for d in val["detail"] if not (d["drop"] or d["review"])]
    if missed:
        print("MISSED noise claims (review and add rules if needed):")
        for d in missed:
            print(f"  {d['sample_id']} patterns={d['noise_patterns']} prop={d['property']!r}")

    # Random spot-check sample for precision (heuristic + gpt-4.1-mini)
    print(f"\n=== Stratified random sample for precision spot-check (heuristic + gpt-4.1-mini) ===")
    samp = stratified_random_sample("annotation_gpt4.1mini_heuristic", k_per_bin=50, seed=0)
    rows = []
    n_drop = n_review = n_keep = 0
    for r in samp:
        out = evaluate_claim(r.get("property",""), r.get("text",""))
        if out["drop"]: n_drop += 1; status = "DROP"
        elif out["review"]: n_review += 1; status = "REVIEW"
        else: n_keep += 1; status = "KEEP"
        rows.append({
            "score": r["_score"],
            "dataset": r["_dataset"],
            "status": status,
            "rules": ";".join(out["rules_fired"]),
            "coverage": round(out["coverage"], 3),
            "property": r.get("property",""),
            "source_text_preview": (r.get("text","") or "")[:200].replace("\n", " "),
        })
    print(f"  total: {len(samp)}, drop: {n_drop}, review: {n_review}, keep: {n_keep}")

    # Write outputs
    (OUT / "filter_validation.json").write_text(json.dumps({
        "recall_drop_only": val["recall_drop_only"],
        "recall_any": val["recall_any"],
        "noise_sample_size": val["noise_sample_size"],
        "rule_hits_on_noise_sample": dict(rule_hits),
        "noise_sample_detail": val["detail"],
        "spot_check_sample_summary": {
            "n_total": len(samp), "n_drop": n_drop, "n_review": n_review, "n_keep": n_keep,
        },
    }, indent=2))
    out_csv = OUT / "random_drop_review.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows: w.writerow(r)
    print(f"\nWrote {OUT/'filter_validation.json'} and {out_csv}")

if __name__ == "__main__":
    main()
