#!/usr/bin/env python3
"""
Alternative-annotator robustness test (Plan B) for Stage B Round 4 (tricky scores 3 & 7).

Question it answers:
  (B1) Is the score-3 vs score-7 strict-pass GAP robust to WHICH annotators labeled it?
       -> leave-one-annotator-out (LOAO) on label-level pooled pass rates + Krippendorff alpha.
  (B2) Is a given annotator (default: the new annotator, Heejin) IN-DISTRIBUTION?
       -> interchangeability (pairwise agreement target-vs-rest) + leniency profile.

Design notes / why these choices:
  - 3 raters/row, no guaranteed triple overlap -> use LABEL-LEVEL pooled rates, not majority.
  - CIs are CLUSTER-bootstrapped by row_id (the 2-3 labels in a row are correlated).
  - Agreement uses Krippendorff alpha (handles variable raters per unit + missing), with
    mean pairwise agreement reported alongside for comparability with the round-3 summary.
  - Runs on whatever Annotation_<name>_new_annot.csv files are present, so it works before
    all six return and scales automatically as more arrive.

Outputs (in OUTDIR, one block per scope):
  overlap_report.csv      coverage / co-labeling structure
  annotator_leniency.csv  per-annotator strictness profile
  passrates_by_score.csv  score 3 vs 7 strict/lenient pass + bootstrap CI + gap
  loao_gap.csv            leave-one-annotator-out gap (delta = rate7 - rate3)
  loao_alpha.csv          Krippendorff alpha per field, full + each LOAO
  target_vs_rest.csv      new-annotator interchangeability check
  summary.md              human-readable synthesis + verdicts + caveats

Scopes:
  round4_new  the 42 designed tricky rows (PRIMARY headline)
  all37       round4_new + round3_old_missing score-3/7 rows (max overlap; sensitivity only)
"""
from __future__ import annotations
import csv, argparse, random, datetime
from pathlib import Path
from collections import defaultdict, Counter
from itertools import combinations

HERE = Path(__file__).resolve().parent
KEY = HERE / "combined" / "combined_assignment_key.csv"
ANNOTATORS = ["Yang", "Yu", "Alexander", "Jingwei", "Tobi", "Heejin"]
FILEPAT = lambda name: HERE / f"Annotation_{name}_new_annot.csv"
RAW = {"cwf": "claim_well_formed", "ent": "entailment", "sc": "score_correct"}
SEED = 12345

csv.field_size_limit(2**31 - 1)


# ---------------------------------------------------------------- cleaning
def clean(field, v):
    """Mirror round-3 normalization."""
    v = (v or "").strip().lower().replace("ye s", "yes")
    if v in ("?", "n/a", "na"):
        return ""
    if field == "cwf" and v == "partial":   # round-3 rule: partial cwf -> blank
        return ""
    return v


def passes(cwf, sc, lenient=False):
    """Per-label pass. None = not judged (cwf or sc blank)."""
    if cwf != "yes" or sc == "":
        return None
    return sc in (("yes", "partial") if lenient else ("yes",))


# ---------------------------------------------------------------- load
def load(infer_sc=False):
    pack = {}
    for r in csv.DictReader(open(KEY)):
        pack[(r["annotator"], r["row_id"])] = r["pack"]

    rows, present, missing = [], [], []
    for ann in ANNOTATORS:
        fp = FILEPAT(ann)
        if not fp.exists():
            missing.append(ann)
            continue
        present.append(ann)
        for r in csv.DictReader(open(fp)):
            cwf = clean("cwf", r[RAW["cwf"]])
            ent = clean("ent", r[RAW["ent"]])
            sc = clean("sc", r[RAW["sc"]])
            if cwf == "" and ent == "" and sc == "":
                continue  # untouched row
            if infer_sc and sc == "" and cwf == "yes" and ent != "":
                try:
                    s = int(r.get("score_annotator") or "")
                except ValueError:
                    s = None
                if ent == "yes" and s and 8 <= s <= 9:
                    sc = "yes"
                elif ent == "no" and s and 1 <= s <= 3:
                    sc = "yes"
                elif ent == "partial" and s and 4 <= s <= 7:
                    sc = "yes"
                else:
                    sc = "no"
            rows.append(dict(
                row_id=r["row_id"], annotator=ann,
                pack=pack.get((ann, r["row_id"]), "?"),
                score=(r.get("score_annotator") or "").strip(),
                combo=r.get("combo", ""), dataset=r.get("dataset", ""),
                cwf=cwf, ent=ent, sc=sc,
                ps=passes(cwf, sc, False), pl=passes(cwf, sc, True),
            ))
    return rows, present, missing


def scoped(rows, scope):
    if scope == "round4_new":
        return [r for r in rows if r["pack"] == "round4_new"]
    if scope == "all37":
        return [r for r in rows if r["score"] in ("3", "7")]
    return rows


# ---------------------------------------------------------------- metrics
def leniency(rows, present):
    out = []
    for ann in present:
        rr = [r for r in rows if r["annotator"] == ann]
        judged = [r for r in rr if r["ps"] is not None]
        sp = sum(1 for r in judged if r["ps"])
        out.append(dict(
            annotator=ann, n=len(rr), n_judged=len(judged), strict_pass=sp,
            strict_rate=round(sp / len(judged), 3) if judged else "",
            sc_yes=sum(1 for r in rr if r["sc"] == "yes"),
            sc_no=sum(1 for r in rr if r["sc"] == "no"),
            ent_partial=sum(1 for r in rr if r["ent"] == "partial"),
            cwf_no=sum(1 for r in rr if r["cwf"] == "no"),
        ))
    return out


def rate_by_score(rows, lenient=False):
    res = {}
    for s in ("3", "7"):
        rr = [r for r in rows if r["score"] == s and (r["pl"] if lenient else r["ps"]) is not None]
        n = len(rr)
        p = sum(1 for r in rr if (r["pl"] if lenient else r["ps"]))
        res[s] = (p, n, (p / n if n else None))
    return res


def bootstrap(rows, B, lenient, seed=SEED):
    """Cluster bootstrap by row_id, stratified by score. Returns CI for rate3, rate7, delta."""
    rng = random.Random(seed)
    byscore = defaultdict(lambda: defaultdict(list))
    for r in rows:
        v = r["pl"] if lenient else r["ps"]
        if v is None:
            continue
        byscore[r["score"]][r["row_id"]].append(1 if v else 0)
    ids = {s: list(byscore[s].keys()) for s in ("3", "7")}
    r3, r7, dd = [], [], []
    for _ in range(B):
        def rate(s):
            pool = ids[s]
            if not pool:
                return None
            pick = [rng.choice(pool) for _ in pool]
            vals = [v for rid in pick for v in byscore[s][rid]]
            return (sum(vals) / len(vals)) if vals else None
        a, b = rate("3"), rate("7")
        if a is not None:
            r3.append(a)
        if b is not None:
            r7.append(b)
        if a is not None and b is not None:
            dd.append(b - a)

    def ci(xs):
        if len(xs) < 2:
            return ("", "")
        xs = sorted(xs)
        lo = xs[int(0.025 * len(xs))]
        hi = xs[min(len(xs) - 1, int(0.975 * len(xs)))]
        return (round(lo, 3), round(hi, 3))
    return ci(r3), ci(r7), ci(dd)


def _field_values_by_row(rows, field):
    """row_id -> list of non-missing values for `field` (ps mapped to P/F)."""
    by = defaultdict(list)
    for r in rows:
        if field == "ps":
            v = "" if r["ps"] is None else ("P" if r["ps"] else "F")
        else:
            v = r[field]
        if v:
            by[r["row_id"]].append(v)
    return by


ORDER = {"ent": ["no", "partial", "yes"], "sc": ["no", "partial", "yes"]}  # ordinal scale


def krippendorff(rows, field, level="nominal", order=None):
    """Krippendorff alpha. level='nominal' (default) or 'ordinal' (needs `order`).

    Ordinal metric: delta^2(c,k) = ( sum_{g in [c..k]} n_g - (n_c + n_k)/2 )^2,
    so adjacent-category disagreement (yes vs partial) is penalized far less than
    far disagreement (yes vs no). Nominal penalizes any mismatch equally.
    """
    units = [v for v in _field_values_by_row(rows, field).values() if len(v) >= 2]
    if not units:
        return None, 0
    o = defaultdict(float)
    n = 0.0
    vals = set()
    for u in units:
        m = len(u)
        cnt = Counter(u)
        vals |= set(cnt)
        for c in cnt:
            for k in cnt:
                if c == k:
                    o[(c, k)] += cnt[c] * (cnt[c] - 1) / (m - 1)
                else:
                    o[(c, k)] += cnt[c] * cnt[k] / (m - 1)
        n += m
    nc = {c: sum(o[(c, k)] for k in vals) for c in vals}

    if level == "ordinal" and order:
        rank = {v: i for i, v in enumerate(order)}
        if any(v not in rank for v in vals):     # unexpected value -> fall back to nominal
            level = "nominal"

    if level == "ordinal" and order:
        def d2(c, k):
            lo, hi = sorted((rank[c], rank[k]))
            between = sum(nc[g] for g in vals if lo <= rank[g] <= hi)
            s = between - (nc[c] + nc[k]) / 2.0
            return s * s
    else:
        def d2(c, k):
            return 0.0 if c == k else 1.0

    Do = sum(o[(c, k)] * d2(c, k) for c in vals for k in vals)
    De = sum(nc[c] * nc[k] * d2(c, k) for c in vals for k in vals)
    if De == 0:
        return None, len(units)          # no expected disagreement (degenerate)
    return round(1 - (n - 1) * Do / De, 3), len(units)


def pairwise_agreement(rows, field):
    by = _field_values_by_row(rows, field)
    agree = tot = 0
    for vals in by.values():
        for a, b in combinations(vals, 2):
            tot += 1
            agree += (a == b)
    return (round(agree / tot, 3) if tot else None), tot


def target_vs_rest(rows, target, field):
    by = defaultdict(list)
    for r in rows:
        if field == "ps":
            v = "" if r["ps"] is None else ("P" if r["ps"] else "F")
        else:
            v = r[field]
        if v:
            by[r["row_id"]].append((r["annotator"], v))
    ta = tt = oa = ot = 0
    for vals in by.values():
        for (a1, v1), (a2, v2) in combinations(vals, 2):
            if target in (a1, a2):
                tt += 1; ta += (v1 == v2)
            else:
                ot += 1; oa += (v1 == v2)
    return ((round(ta / tt, 3) if tt else None), tt), ((round(oa / ot, 3) if ot else None), ot)


def overlap(rows):
    by = defaultdict(set)
    for r in rows:
        by[r["row_id"]].add(r["annotator"])
    dist = Counter(len(a) for a in by.values())
    pair = Counter()
    rows_for = defaultdict(set)
    for rid, anns in by.items():
        for a, b in combinations(sorted(anns), 2):
            pair[(a, b)] += 1
    return dist, pair, sum(1 for a in by.values() if len(a) >= 2)


# ---------------------------------------------------------------- io
def fci(t):
    return f"{t[0]}-{t[1]}" if t and t[0] != "" else ""


def write_csv(path, rows, fields):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def run_scope(allrows, present, scope, outdir, B, target):
    rows = scoped(allrows, scope)
    tag = scope
    lines = [f"\n## Scope: `{scope}`  (labels={len(rows)})\n"]

    # overlap
    dist, pair, n_co = overlap(rows)
    write_csv(outdir / f"overlap_report__{tag}.csv",
              [dict(metric="labels_per_row=%d" % k, value=v) for k, v in sorted(dist.items())]
              + [dict(metric=f"overlap[{a}&{b}]", value=v) for (a, b), v in sorted(pair.items())],
              ["metric", "value"])
    lines.append(f"- co-labeled rows (>=2 labels): **{n_co}**; labels-per-row dist: "
                 + ", ".join(f"{k}->{v}" for k, v in sorted(dist.items())))
    if pair:
        lines.append("- pairwise overlap: " + ", ".join(f"{a}&{b}={v}" for (a, b), v in sorted(pair.items())))

    # leniency
    len_rows = leniency(rows, present)
    write_csv(outdir / f"annotator_leniency__{tag}.csv", len_rows,
              ["annotator", "n", "n_judged", "strict_pass", "strict_rate", "sc_yes", "sc_no", "ent_partial", "cwf_no"])
    lines.append("- per-annotator strict-pass: "
                 + ", ".join(f"{r['annotator']}={r['strict_rate']}" for r in len_rows))

    # pass rates by score + gap
    pr = []
    for lenient in (False, True):
        rs = rate_by_score(rows, lenient)
        ci3, ci7, cid = bootstrap(rows, B, lenient)
        kind = "lenient" if lenient else "strict"
        gap = (rs["7"][2] - rs["3"][2]) if (rs["3"][2] is not None and rs["7"][2] is not None) else None
        pr += [
            dict(rule=kind, score=3, n_judged=rs["3"][1], n_pass=rs["3"][0],
                 rate=round(rs["3"][2], 3) if rs["3"][2] is not None else "", ci=fci(ci3)),
            dict(rule=kind, score=7, n_judged=rs["7"][1], n_pass=rs["7"][0],
                 rate=round(rs["7"][2], 3) if rs["7"][2] is not None else "", ci=fci(ci7)),
            dict(rule=kind, score="gap(7-3)", n_judged="", n_pass="",
                 rate=round(gap, 3) if gap is not None else "", ci=fci(cid)),
        ]
        if not lenient:
            lines.append(f"- **strict pass**: score3={pr[0]['rate']} {pr[0]['ci']} | "
                         f"score7={pr[1]['rate']} {pr[1]['ci']} | "
                         f"**gap={pr[2]['rate']} {pr[2]['ci']}**")
    write_csv(outdir / f"passrates_by_score__{tag}.csv", pr,
              ["rule", "score", "n_judged", "n_pass", "rate", "ci"])

    # LOAO gap
    loao = []
    variants = [("full", rows)] + [(f"-{a}", [r for r in rows if r["annotator"] != a]) for a in present]
    for name, rr in variants:
        rs = rate_by_score(rr, False)
        ci3, ci7, cid = bootstrap(rr, B, False)
        gap = (rs["7"][2] - rs["3"][2]) if (rs["3"][2] is not None and rs["7"][2] is not None) else None
        loao.append(dict(
            variant=name, n3=rs["3"][1], rate3=round(rs["3"][2], 3) if rs["3"][2] is not None else "",
            n7=rs["7"][1], rate7=round(rs["7"][2], 3) if rs["7"][2] is not None else "",
            delta=round(gap, 3) if gap is not None else "", delta_ci=fci(cid)))
    write_csv(outdir / f"loao_gap__{tag}.csv", loao,
              ["variant", "n3", "rate3", "n7", "rate7", "delta", "delta_ci"])
    deltas = [r["delta"] for r in loao if r["delta"] != ""]
    if deltas:
        signstable = all((d > 0) == (deltas[0] > 0) for d in deltas)
        full_d = loao[0]["delta"]
        big = [r["variant"] for r in loao[1:] if r["delta"] != "" and full_d != "" and abs(r["delta"] - full_d) > 0.5 * abs(full_d)]
        lines.append(f"- **LOAO**: delta range [{min(deltas)}, {max(deltas)}], "
                     f"sign-stable={signstable}; drops that move delta >50%: {big or 'none'}")

    # Krippendorff alpha + pairwise, full and LOAO
    alpha = []
    for name, rr in variants:
        row = {"variant": name}
        for f in ("cwf", "ent", "sc", "ps"):
            a, nu = krippendorff(rr, f, "nominal")
            pa, npr = pairwise_agreement(rr, f)
            row[f + "_alpha"] = a if a is not None else ""
            row[f + "_units"] = nu
            row[f + "_pair_agr"] = pa if pa is not None else ""
            if f in ORDER:                        # ordinal variant for ent/sc
                ao, _ = krippendorff(rr, f, "ordinal", ORDER[f])
                row[f + "_alpha_ord"] = ao if ao is not None else ""
        alpha.append(row)
    write_csv(outdir / f"loao_alpha__{tag}.csv", alpha,
              ["variant", "cwf_alpha", "cwf_units", "cwf_pair_agr",
               "ent_alpha", "ent_alpha_ord", "ent_units", "ent_pair_agr",
               "sc_alpha", "sc_alpha_ord", "sc_units", "sc_pair_agr",
               "ps_alpha", "ps_units", "ps_pair_agr"])
    full_alpha = alpha[0]
    lines.append("- agreement (full): "
                 + ", ".join(f"{f} alpha={full_alpha[f+'_alpha']}"
                             + (f"/ord={full_alpha[f+'_alpha_ord']}" if f in ORDER else "")
                             + f"/pair={full_alpha[f+'_pair_agr']}(n={full_alpha[f+'_units']})"
                             for f in ("cwf", "ent", "sc", "ps")))

    # target (new-annotator) check
    tvr = []
    for f in ("cwf", "ent", "sc", "ps"):
        (ta, tt), (oa, ot) = target_vs_rest(rows, target, f)
        tvr.append(dict(field=f, target=target,
                        target_pair_agr=ta if ta is not None else "", target_pairs=tt,
                        others_pair_agr=oa if oa is not None else "", other_pairs=ot))
    write_csv(outdir / f"target_vs_rest__{tag}.csv", tvr,
              ["field", "target", "target_pair_agr", "target_pairs", "others_pair_agr", "other_pairs"])
    if target in present:
        tr = leniency(rows, [target])
        tl = tr[0]["strict_rate"] if tr else ""
        lines.append(f"- **{target} check**: strict_rate={tl}; "
                     + ", ".join(f"{r['field']}: {target}={r['target_pair_agr']}(n={r['target_pairs']}) "
                                 f"others={r['others_pair_agr']}(n={r['other_pairs']})" for r in tvr))
    else:
        lines.append(f"- {target} not present yet -> new-annotator check skipped.")

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--target", default="Heejin", help="annotator to run the in-distribution check on")
    ap.add_argument("--infer-sc", action="store_true", help="round-3 fallback: infer blank sc from ent+score")
    args = ap.parse_args()

    outdir = Path(args.outdir) if args.outdir else HERE / f"analysis_altannot_{datetime.date.today().isoformat()}"
    outdir.mkdir(parents=True, exist_ok=True)

    rows, present, missing = load(infer_sc=args.infer_sc)
    header = [
        "# Round-4 Alternative-Annotator Robustness Test (Plan B)",
        f"\n- annotators present: **{present or 'none'}**",
        f"- annotators missing: **{missing or 'none'}**",
        f"- total labels loaded: {len(rows)}  | bootstrap B={args.bootstrap} | infer_sc={args.infer_sc}",
        "\n> Caveats: label-level (not majority) rates; CIs cluster-bootstrapped by row_id; "
        "alpha is nominal Krippendorff. With <6 annotators or thin overlap, LOAO is a "
        "'does one rater drive it' probe, NOT full generalizability. This is the worst-case "
        "(tricky score-3/7) subset; fragility here does not condemn the easy bins.",
    ]
    blocks = [run_scope(rows, present, sc, outdir, args.bootstrap, args.target)
              for sc in ("round4_new", "all37")]
    (outdir / "summary.md").write_text("\n".join(header) + "\n" + "\n".join(blocks) + "\n")
    print(f"wrote outputs to {outdir}")
    print("\n".join(header))
    for b in blocks:
        print(b)


if __name__ == "__main__":
    main()
