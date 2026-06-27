# Stage A — Programmatic Audit Summary

Outputs:
- `summary.json` — top-level numbers (this file's source).
- `per_raw_combo/<combo>.json` — full per-combo metrics for the 7 raw `(parser, annotator)` combos.
- `per_experiment_branch/<branch>.json` — per-branch metrics for the 34 trained-on branches.
- `A5_cross_parser_overlap.json` — pairwise source-text overlap.

The actionable findings are on the **raw `experiments/` layer (Layer 2)**. Layer 1 numbers (split sanity, rebalancing magnitudes) are reported at the end as confirmation, since splits and rebalancing don't exist in the raw layer.

## A. Raw-data findings (the ones that matter)

### A.1 Score-9 imbalance has both a parser effect and an annotator effect, and they compound

Layer 2 score distribution by bin (1–4 / 5–7 / 8–9):

| Combo | 1–4 | 5–7 | 8–9 | Rows |
|---|---:|---:|---:|---:|
| heuristic + gpt-4.1-mini | 17.8% | 14.6% | **67.6%** | 22 479 |
| heuristic + gpt-5-mini | 11.3% | 4.6% | **84.1%** | 22 448 |
| heuristic + gpt-oss | 18.0% | 15.6% | **66.4%** | 1 500 |
| llm gpt-5-mini + gpt-4.1-mini | 4.5% | 6.6% | **88.9%** | 19 495 |
| llm gpt-5-mini + gpt-5-mini | 3.6% | 1.4% | **95.0%** | 19 431 |
| llm gpt-oss + gpt-4.1-mini | 6.4% | 10.5% | **83.2%** | 22 551 |
| llm gpt-oss + gpt-oss | 6.3% | 9.3% | **84.4%** | 1 500 |

Compact summary at the same gpt-4.1-mini annotator:

| | gpt-4.1-mini ann | gpt-5-mini ann |
|---|---:|---:|
| heuristic parser | 68% at score 9 | 84% |
| llm gpt-5-mini parser | **89%** | **95%** |

Same ~1 470 source articles (Jaccard ~0.99 across these combos, see A.4), so this is not corpus drift. **Hypothesis I (imbalance is partly a parser artefact) is supported.** Whether it survives filtering is for Stage A.5 to determine.

### A.2 Parser-noise patterns differ qualitatively (not just in degree)

| Failure mode | heuristic | LLM gpt-5-mini | LLM gpt-oss |
|---|---:|---:|---:|
| starts "the text" prefix | 93–95% | 96% | 97–99% |
| `the text uses the phrase ...` (substring detection) | **13.5%** | 0.3% | 0.5–0.7% |
| Reasoning leakage (`i should`, `let me` …) | **1.5%** | 0.0% | 0.0% |
| token length < 5 (fragments) | 1.1–1.4% | 0.5% | 0.7–1.5% |
| **coverage = 0** (claim content not in source) | 26–28% | 25% | **33%** |
| exact-property duplicate rate | 1.0–5.0% | 3.9% | 1.5–8.4% |

Two qualitatively distinct noise modes:
- **Heuristic-specific**: substring detection (`the text uses the phrase ...`) and reasoning-leakage fragments. Visible, easy to filter with regex.
- **LLM-specific (and worse for gpt-oss)**: claim content not in source — "plausible-sounding text attached to the wrong place". Harder to spot without a coverage check.

Implication: heuristic isn't unambiguously noisier — it has *more visible* noise but the LLM parsers have a comparable rate of misalignment-style noise. The Stage A.5 filter needs to cover both kinds (regex rules for heuristic templates, coverage rule for LLM misalignment).

### A.3 Score parsing is reliable; Layer 1 ↔ Layer 2 is faithful (A7)

- `[Answer]: N` regex resolves to a 1–9 integer for ≥ 99.6% of rows; `score_extract_failures` are 0–96 out of ~22 k per combo.
- 0 score mismatches across all 7 combos when comparing rounded `accuracy_score` in Layer 1 against the parsed integer in Layer 2 `output`. Layer 1 faithfully tracks Layer 2.
- 32–225 rows present in Layer 2 but not in Layer 1 (5 main combos), 0 for the gpt-oss-annotator probes. Most likely deduplication during split construction. **No silent data loss.**
- **Cosmetic note**: gpt-oss-annotated branches store `accuracy_score` as float-precision-noisy values (e.g. `8.999999999886083`). Other combos store exact integers. If any pipeline code uses `int()` instead of `round()`, gpt-oss labels will silently truncate by 1.
- Annotator `reasoning_content` is non-empty for the gpt-4.1-mini and gpt-5-mini annotators; often empty for gpt-oss. Stage B should rely on `reasoning_content` only for the non-gpt-oss combos.

### A.4 Cross-parser source-text overlap (A5) — paired comparison is well-grounded

| Pair (gpt-4.1-mini annotator) | Jaccard | Shared text_ids |
|---|---:|---:|
| heuristic ↔ llm gpt-5-mini | 0.991 | 1 469 |
| heuristic ↔ llm gpt-oss | 1.000 | 1 482 |
| llm gpt-5-mini ↔ llm gpt-oss | 0.991 | 1 469 |
| any-main ↔ gpt-oss-annotator probes | ~0.075 | ~111 |

The 5 main combos sit on essentially the same set of ~1 470 source articles. Stage B paired-article comparison (30 articles) will draw cleanly from this overlap. The two gpt-oss-annotator probe combos sit on a much smaller and largely separate ~111-article corpus and should be reported descriptively, not as headline branches.

## B. Layer 1 confirmatory checks

Reported here for completeness — these are not raw-data audit findings, they're properties of the splits and rebalancing routines downstream of the raw data.

### B.1 Splits are clean (A1)

All 34 experiment branches have **0.00% `text_id` leakage** and **0.00% exact-text leakage** across train / val / test. The split procedure was done correctly at the source-article level. **No prior modelling result needs to be redone for split reasons.**

### B.2 Rebalancing magnitudes (A6) — not new information, but worth reading off

The `reg_over_exact` and `reg_down_exact` rebalancing routines produce mechanically-pathological train sets:

| Branch | Train | Dup rate | Max single-`id` rep |
|---|---:|---:|---:|
| heuristic + gpt-4.1-mini, reg_over_exact | 74 538 | **79.9%** | 898× |
| llm gpt-5-mini + gpt-4.1-mini, reg_over_exact | 98 082 | **86.6%** | **5 566×** |
| llm gpt-5-mini + gpt-5-mini, reg_over_exact | 112 617 | **88.4%** | 554× |
| heuristic + gpt-5-mini, reg_over_exact | 111 942 | **86.5%** | 124× |
| heuristic + gpt-4.1-mini, reg_down_exact | **90** | 0.0% | 1× |
| llm gpt-5-mini + gpt-4.1-mini, reg_down_exact | **18** | 0.0% | 1× |

These slices will be **dropped at Stage D** under the train-size and dup-rate rules in `audit_decision_rules.md` §1b. `reg_base`, `reg_down_bins`, and `reg_over_bins` look fine for modelling.

This isn't a "discovery" — oversampling is *expected* to duplicate rare-class rows. The magnitudes (especially 5 566×) are useful for re-interpreting prior modelling results on those slices: anything reported on `reg_over_exact` is closer to memorisation than to generalisation.

## C. Headline takeaways

1. **Raw-data noise is the real audit target.** Heuristic has a distinct, regex-tractable failure mode (substring detection, reasoning leakage); LLM parsers have a roughly equal rate (25–33%) of coverage = 0 misalignment which is harder to spot but still filterable.
2. **Score-9 imbalance scales with both parser switch (heuristic → LLM) and annotator switch (gpt-4.1-mini → gpt-5-mini)**, on the same source articles. Hypothesis I supported; Stage A.5 will tell us whether filtering closes the gap.
3. **Splits are clean** — no `text_id` leakage anywhere.
4. **Layer 1 faithfully tracks Layer 2** — no silent data loss; one float-precision quirk for gpt-oss labels.
5. **`reg_over_exact` and `reg_down_exact` are pathological by construction** and will be dropped at Stage D — confirming what we already suspected.

## D. Filter targets for Stage A.5

Drawn from A.2:

- **Templated noise** (regex, all anchored at start after the "the text" prefix):
  - `^the text uses the phrase`
  - `^the text is (a tweet|relevant|irrelevant)\.?$`
  - `^the text the (article|possible)\b` (stutter pattern)
  - `^the text does not (mention|reference|discuss|contain|include)` paired with coverage = 0
- **Reasoning leakage** (regex at start): `i should`, `let me`, `looking at`, `without (knowing|more)`, `i need to`, `i would`
- **Length / shape**: token count < 5 (after stripping prefix), or > 50, or ends with `?`
- **Source grounding**: `coverage = 0` → auto-drop; `coverage in (0, 0.10]` → flag for review
- **Within-`text_id` exact-property duplicates**: keep one, drop the rest

Validation against `experiments/analysis/noisy_heuristic_claims_three_models_sample_20.csv` (target precision ≥ 0.85, recall ≥ 0.70).
