"""Stage A — programmatic audit. See data_inspection_plan_v2.md.

Outputs JSON per raw combo and per experiment branch. No manual work.
Runs in stdlib only (no extra deps required).
"""
from __future__ import annotations
import csv, json, os, re, hashlib, statistics, sys
from collections import Counter, defaultdict
from pathlib import Path

csv.field_size_limit(2**31 - 1)

ROOT = Path("/Users/yangtian/Documents/law/calib/RuleFollower")
EXP = ROOT / "experiments"
MB = ROOT / "modernbert_uq" / "data"
OUT = ROOT / "modernbert_uq" / "results" / "audit" / "stageA"
OUT_RAW = OUT / "per_raw_combo"
OUT_BR = OUT / "per_experiment_branch"

RAW_COMBOS = [
    ("heuristic_ann_gpt41mini",       "annotation_gpt4.1mini_heuristic",     "heuristic", "gpt-4.1-mini"),
    ("heuristic_ann_gpt5mini",        "annotation_gpt5mini_heuristic",       "heuristic", "gpt-5-mini"),
    ("heuristic_ann_gptoss",          "annotation_gpt-oss_heuristic",        "heuristic", "gpt-oss"),
    ("llm_gpt5mini_ann_gpt41mini",    "annotation_gpt4.1mini_gpt-5-mini",    "llm_gpt5mini", "gpt-4.1-mini"),
    ("llm_gpt5mini_ann_gpt5mini",     "annotation_gpt5mini",                  "llm_gpt5mini", "gpt-5-mini"),
    ("llm_gptoss_ann_gpt41mini",      "annotation_gpt4.1mini_gpt-oss",       "llm_gptoss", "gpt-4.1-mini"),
    ("llm_gptoss_ann_gptoss",         "annotation_gpt-oss",                  "llm_gptoss", "gpt-oss"),
]

ANS_RE = re.compile(r"\[?Answer\]?:?\s*([0-9]+)", re.IGNORECASE)

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

def alpha_words(s: str) -> list[str]:
    return re.findall(r"[A-Za-z][A-Za-z'\-]{2,}", s or "")

def lemma_proxy_coverage(prop: str, source: str) -> float:
    if not prop or not source:
        return 1.0
    toks = [t.lower() for t in alpha_words(prop)
            if t.lower() not in STOPWORDS and t.lower() not in META_TOKENS]
    if not toks:
        return 1.0
    src = source.lower()
    return sum(1 for t in toks if t in src) / len(toks)

# ---------- read helpers ----------
def read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))

def text_id_hash(text: str) -> str:
    return hashlib.md5((text or "").strip().encode("utf-8")).hexdigest()[:16]

def normalise_property(p: str) -> str:
    return re.sub(r"\s+", " ", (p or "").strip().lower())

def safe_p(values, q):
    if not values: return None
    s = sorted(values)
    idx = int(q * (len(s) - 1))
    return s[idx]

# ---------- A3 / A4 / A5 / A7 per raw combo ----------
def audit_raw_combo(combo_name: str, ann_folder: str, parser: str, annotator: str) -> dict:
    folder = EXP / ann_folder / "accuracy_outputs"
    files = sorted(folder.glob("*.csv"))
    rows_per_dataset = {}
    rows_all = []
    for f in files:
        ds = f.name.split("_score_")[0]
        rs = read_csv(f)
        for r in rs:
            r["_dataset"] = ds
        rows_per_dataset[ds] = rs
        rows_all.extend(rs)

    n = len(rows_all)
    # A4: parse score from `output` to a per-row int
    parsed = []
    for r in rows_all:
        m = ANS_RE.search(r.get("output") or "")
        try: s = int(m.group(1)) if m else None
        except: s = None
        r["_score_int"] = s
        if s is not None: parsed.append(s)
    score_dist = Counter(parsed)
    score_dist_full = {s: score_dist.get(s, 0) for s in range(1, 10)}
    score_extract_failures = sum(1 for r in rows_all if r["_score_int"] is None)

    # A3: claim shape stats (on `property`)
    char_lens = [len(r.get("property") or "") for r in rows_all]
    tok_lens  = [len((r.get("property") or "").split()) for r in rows_all]
    tok_lt5 = sum(1 for x in tok_lens if x < 5)
    tok_gt50 = sum(1 for x in tok_lens if x > 50)
    starts_the_text = sum(1 for r in rows_all
                          if (r.get("property") or "").strip().lower().startswith("the text"))
    uses_phrase = sum(1 for r in rows_all
                      if re.match(r"^the text uses the phrase", (r.get("property") or "").strip().lower()))
    stutter = sum(1 for r in rows_all
                  if re.match(r"^the text the (article|possible)\b", (r.get("property") or "").strip().lower()))
    reasoning_leak = sum(1 for r in rows_all
                         if re.match(r"^(?:the text )?(i should|let me|looking at|without (?:knowing|more)|i need to|i would)\b",
                                     (r.get("property") or "").strip().lower()))
    is_question = sum(1 for r in rows_all if (r.get("property") or "").rstrip().endswith("?"))

    # verbatim substring of source (text column = source in Layer 2)
    verbatim = 0
    for r in rows_all:
        p = (r.get("property") or "").strip()
        s = r.get("text") or ""
        if p and s and p in s: verbatim += 1

    # lemma-proxy coverage stats + zero-coverage count
    covs = [lemma_proxy_coverage(r.get("property",""), r.get("text","")) for r in rows_all]
    cov_zero = sum(1 for c in covs if c == 0)
    cov_le_010 = sum(1 for c in covs if 0 < c <= 0.10)

    # Within-corpus exact-property duplication (raw)
    norm_props = [normalise_property(r.get("property") or "") for r in rows_all]
    prop_counts = Counter(norm_props)
    raw_exact_dup_extra = sum(c for c in prop_counts.values() if c > 1) - len([c for c in prop_counts.values() if c > 1])
    top_dups = prop_counts.most_common(15)

    # Conditional length / dup by score bin
    bins = {"1-4": [s for s in parsed if 1 <= s <= 4],
            "5-7": [s for s in parsed if 5 <= s <= 7],
            "8-9": [s for s in parsed if 8 <= s <= 9]}
    bin_counts = {k: len(v) for k,v in bins.items()}
    # length conditional on score
    by_score_len = defaultdict(list)
    for r in rows_all:
        if r["_score_int"] is not None:
            by_score_len[r["_score_int"]].append(len((r.get("property") or "").split()))
    score_len_summary = {s: {"n": len(v), "mean_tokens": (statistics.mean(v) if v else 0)}
                         for s,v in sorted(by_score_len.items())}

    # text_id (hash of source text) for cross-parser overlap (A5)
    tid_set = set()
    for r in rows_all:
        r["_tid"] = text_id_hash(r.get("text") or "")
        tid_set.add(r["_tid"])

    # A7: Layer 1 ↔ Layer 2 reconciliation against the matching reg_base branch
    l1_branch = MB / f"rulefollower_{combo_name}_reg_base"
    if not l1_branch.exists():
        # fall back to ge8
        l1_branch = MB / f"rulefollower_{combo_name}_ge8"
    l1_rows = []
    l1_present = l1_branch.exists()
    if l1_present:
        for split_name in ("train.csv", "validation.csv", "test.csv"):
            p = l1_branch / split_name
            if p.exists():
                for row in read_csv(p):
                    row["_split"] = split_name.replace(".csv","")
                    l1_rows.append(row)
    # Build a key on (text_id_hash from Layer 2) ↔ (text_id from Layer 1) using property+text hash
    l2_key = {}
    for r in rows_all:
        key = (r["_tid"], normalise_property(r.get("property") or ""))
        l2_key.setdefault(key, []).append(r)
    l1_match = 0
    l1_mismatch_score = 0
    l1_score_examples = []
    for r in l1_rows:
        tid_h = text_id_hash(r.get("source_text") or "")
        key = (tid_h, normalise_property(r.get("property") or ""))
        if key in l2_key:
            l1_match += 1
            l2r = l2_key[key][0]
            if l2r["_score_int"] is not None:
                try:
                    # round, not truncate: Layer 1 stores soft floats (e.g. 8.999... ≈ 9)
                    l1_score = int(round(float(r["accuracy_score"])))
                except:
                    l1_score = None
                if l1_score is not None and l1_score != l2r["_score_int"]:
                    l1_mismatch_score += 1
                    if len(l1_score_examples) < 5:
                        l1_score_examples.append({
                            "id": r.get("id"),
                            "l2_output": (l2r.get("output") or "")[:80],
                            "l2_score_parsed": l2r["_score_int"],
                            "l1_accuracy_score": r.get("accuracy_score"),
                        })
    l1_total = len(l1_rows)
    l2_total = n
    drop_into_l1 = l2_total - l1_match  # rows in Layer 2 not present in Layer 1 (this branch's splits)

    # reasoning availability
    rc_nonempty = sum(1 for r in rows_all if (r.get("reasoning_content") or "").strip())

    out = {
        "combo": combo_name,
        "parser": parser,
        "annotator": annotator,
        "experiments_folder": ann_folder,
        "rows_layer2_total": n,
        "score_extract_failures": score_extract_failures,
        "score_distribution_layer2": score_dist_full,
        "score_bin_distribution": bin_counts,
        "claim_shape": {
            "char_len_mean": statistics.mean(char_lens) if char_lens else 0,
            "char_len_median": statistics.median(char_lens) if char_lens else 0,
            "token_len_mean": statistics.mean(tok_lens) if tok_lens else 0,
            "token_len_median": statistics.median(tok_lens) if tok_lens else 0,
            "token_len_p10": safe_p(tok_lens, 0.10),
            "token_len_p90": safe_p(tok_lens, 0.90),
            "fraction_starts_the_text": starts_the_text / n if n else 0,
            "fraction_uses_phrase": uses_phrase / n if n else 0,
            "fraction_stutter_the_text_the": stutter / n if n else 0,
            "fraction_reasoning_leak": reasoning_leak / n if n else 0,
            "fraction_question_mark": is_question / n if n else 0,
            "fraction_verbatim_substring_of_source": verbatim / n if n else 0,
            "fraction_token_lt5": tok_lt5 / n if n else 0,
            "fraction_token_gt50": tok_gt50 / n if n else 0,
            "fraction_coverage_zero": cov_zero / n if n else 0,
            "fraction_coverage_0_10": cov_le_010 / n if n else 0,
        },
        "score_conditional_token_len_mean": score_len_summary,
        "exact_dup_extra_copies": raw_exact_dup_extra,
        "exact_dup_rate": raw_exact_dup_extra / n if n else 0,
        "top_repeated_properties": [{"prop": p[:160], "count": c} for p,c in top_dups],
        "unique_text_ids": len(tid_set),
        "rows_per_dataset": {k: len(v) for k,v in rows_per_dataset.items()},
        "reconciliation_layer1": {
            "layer1_branch_used": l1_branch.name if l1_present else None,
            "layer1_present": l1_present,
            "layer1_total_rows": l1_total,
            "layer1_match_to_layer2": l1_match,
            "rows_in_layer2_not_in_layer1": drop_into_l1,
            "score_mismatch_count": l1_mismatch_score,
            "score_mismatch_examples": l1_score_examples,
        },
        "reasoning_content_nonempty": rc_nonempty,
        "reasoning_content_nonempty_rate": rc_nonempty / n if n else 0,
    }
    return out, tid_set

# ---------- A1 / A2 / A6 per experiment branch ----------
def audit_experiment_branch(branch_dir: Path) -> dict:
    name = branch_dir.name
    summary_p = branch_dir / "summary.json"
    sumj = json.loads(summary_p.read_text()) if summary_p.exists() else {}

    splits = {}
    for s in ("train", "validation", "test"):
        p = branch_dir / f"{s}.csv"
        if p.exists():
            splits[s] = read_csv(p)
        else:
            splits[s] = []
    # A1 split sanity
    text_to_split = defaultdict(set)
    tid_to_split = defaultdict(set)
    for split_name, rows in splits.items():
        for r in rows:
            text_to_split[r.get("text","").strip()].add(split_name)
            tid_to_split[r.get("text_id","")].add(split_name)
    # exact text leakage = number of distinct texts present in >1 split
    exact_text_leak = sum(1 for v in text_to_split.values() if len(v) > 1)
    # text_id leakage
    tid_leak = sum(1 for v in tid_to_split.values() if len(v) > 1)
    train_rows = len(splits["train"])
    text_leak_pct = exact_text_leak / train_rows if train_rows else 0
    tid_leak_pct = tid_leak / train_rows if train_rows else 0

    # A2 within-train duplication
    train_props_norm = [normalise_property(r.get("property") or "") for r in splits["train"]]
    train_text_norm  = [normalise_property(r.get("text") or "") for r in splits["train"]]
    train_prop_dup = sum(c for c in Counter(train_props_norm).values() if c > 1) - len([c for c in Counter(train_props_norm).values() if c > 1])
    train_text_dup = sum(c for c in Counter(train_text_norm).values() if c > 1) - len([c for c in Counter(train_text_norm).values() if c > 1])
    # Per-source-article fan-out
    fan = Counter()
    for r in splits["train"]:
        fan[r.get("text_id","")] += 1
    fan_vals = list(fan.values())
    fan_summary = {
        "claims_per_article_mean": statistics.mean(fan_vals) if fan_vals else 0,
        "claims_per_article_median": statistics.median(fan_vals) if fan_vals else 0,
        "claims_per_article_max": max(fan_vals) if fan_vals else 0,
        "n_articles_in_train": len(fan_vals),
    }

    # A6 rebalancing sanity (read from summary.json; compute repetition by id)
    id_counts = Counter(r.get("id","") for r in splits["train"])
    max_id_rep = max(id_counts.values()) if id_counts else 0
    rep_gt5 = sum(1 for c in id_counts.values() if c > 5)
    # bin distributions
    train_score_bins = Counter()
    test_score_bins = Counter()
    for r in splits["train"]:
        b = r.get("score_bin")
        if b: train_score_bins[b] += 1
    for r in splits["test"]:
        b = r.get("score_bin")
        if b: test_score_bins[b] += 1
    # pct mismatch
    train_total = sum(train_score_bins.values()) or 1
    test_total = sum(test_score_bins.values()) or 1
    bin_mismatch = {}
    for k in set(train_score_bins) | set(test_score_bins):
        tr = train_score_bins.get(k, 0) / train_total
        te = test_score_bins.get(k, 0) / test_total
        bin_mismatch[k] = {"train_pct": tr, "test_pct": te, "abs_gap": abs(tr-te)}

    return {
        "branch": name,
        "split_sizes": {k: len(v) for k,v in splits.items()},
        "summary_json_rows": sumj.get("rows"),
        "split_sanity": {
            "exact_text_leak_distinct_texts_in_multi_splits": exact_text_leak,
            "text_id_leak_distinct_ids_in_multi_splits": tid_leak,
            "text_leak_pct_of_train": text_leak_pct,
            "text_id_leak_pct_of_train": tid_leak_pct,
        },
        "within_train_duplication": {
            "extra_property_dup_rows": train_prop_dup,
            "extra_text_dup_rows": train_text_dup,
            "property_dup_rate": train_prop_dup / train_rows if train_rows else 0,
            "text_dup_rate": train_text_dup / train_rows if train_rows else 0,
            **fan_summary,
        },
        "rebalancing_sanity": {
            "train_rows": train_rows,
            "max_id_repetition": max_id_rep,
            "n_ids_repeated_gt5": rep_gt5,
            "train_bin_distribution_pct": {k: v/train_total for k,v in train_score_bins.items()},
            "test_bin_distribution_pct": {k: v/test_total for k,v in test_score_bins.items()},
            "bin_train_test_gap": bin_mismatch,
        },
    }

def main():
    print(f"[Stage A] writing outputs to {OUT}")
    OUT_RAW.mkdir(parents=True, exist_ok=True)
    OUT_BR.mkdir(parents=True, exist_ok=True)

    # Per raw combo
    raw_results = {}
    tid_sets = {}
    for combo_name, ann_folder, parser, annotator in RAW_COMBOS:
        if not (EXP / ann_folder / "accuracy_outputs").exists():
            print(f"  [skip] {combo_name}: folder missing")
            continue
        print(f"  [raw] {combo_name} ...", flush=True)
        result, tids = audit_raw_combo(combo_name, ann_folder, parser, annotator)
        (OUT_RAW / f"{combo_name}.json").write_text(json.dumps(result, indent=2))
        raw_results[combo_name] = result
        tid_sets[combo_name] = tids

    # A5 cross-parser source overlap
    overlap = {}
    keys = list(tid_sets.keys())
    for i, a in enumerate(keys):
        for b in keys[i+1:]:
            inter = tid_sets[a] & tid_sets[b]
            overlap[f"{a} ∩ {b}"] = {
                "size": len(inter),
                "a_size": len(tid_sets[a]),
                "b_size": len(tid_sets[b]),
                "jaccard": len(inter) / len(tid_sets[a] | tid_sets[b]) if (tid_sets[a] | tid_sets[b]) else 0,
            }
    (OUT / "A5_cross_parser_overlap.json").write_text(json.dumps(overlap, indent=2))

    # Per experiment branch
    branch_results = {}
    branch_dirs = sorted([d for d in MB.iterdir() if d.is_dir() and not d.name.startswith("misinfo_")])
    for d in branch_dirs:
        if not (d/"train.csv").exists():
            continue
        print(f"  [branch] {d.name} ...", flush=True)
        r = audit_experiment_branch(d)
        (OUT_BR / f"{d.name}.json").write_text(json.dumps(r, indent=2))
        branch_results[d.name] = r

    # Top-level summary
    summary = {
        "n_raw_combos": len(raw_results),
        "n_branches": len(branch_results),
        "raw_summary": {k: {
            "rows_layer2_total": v["rows_layer2_total"],
            "fraction_starts_the_text": v["claim_shape"]["fraction_starts_the_text"],
            "fraction_uses_phrase": v["claim_shape"]["fraction_uses_phrase"],
            "fraction_reasoning_leak": v["claim_shape"]["fraction_reasoning_leak"],
            "fraction_token_lt5": v["claim_shape"]["fraction_token_lt5"],
            "fraction_coverage_zero": v["claim_shape"]["fraction_coverage_zero"],
            "exact_dup_rate": v["exact_dup_rate"],
            "score_extract_failures": v["score_extract_failures"],
            "score_bin_distribution": v["score_bin_distribution"],
            "reconciliation": {
                "layer1_branch_used": v["reconciliation_layer1"]["layer1_branch_used"],
                "rows_in_layer2_not_in_layer1": v["reconciliation_layer1"]["rows_in_layer2_not_in_layer1"],
                "score_mismatch_count": v["reconciliation_layer1"]["score_mismatch_count"],
            },
        } for k,v in raw_results.items()},
        "branch_summary_brief": {k: {
            "train_rows": v["rebalancing_sanity"]["train_rows"],
            "tid_leak_pct": v["split_sanity"]["text_id_leak_pct_of_train"],
            "exact_text_leak_pct": v["split_sanity"]["text_leak_pct_of_train"],
            "property_dup_rate": v["within_train_duplication"]["property_dup_rate"],
            "max_id_rep": v["rebalancing_sanity"]["max_id_repetition"],
        } for k,v in branch_results.items()},
        "cross_parser_overlap": overlap,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[Stage A] done — {len(raw_results)} raw combos, {len(branch_results)} branches.")

if __name__ == "__main__":
    main()
