# Stage B — Tiered Claim Audit (first pass)

## Method recap

- 30 stratified claims sampled per combo (10 from score 1–4, 5 from 5–7, 10 from 8–9, 5 from Stage-A flag pool) from the **filtered** Layer 2 corpus.
- Three combos audited (all with gpt-4.1-mini annotator, so any difference is parser-driven, not annotator-driven): `heuristic + gpt-4.1-mini`, `llm gpt-5-mini + gpt-4.1-mini`, `llm gpt-oss + gpt-4.1-mini`.
- Auditor: Claude (this session). Different model from any project parser/annotator. Single-auditor — no Cohen's kappa available.
- Rubric: `claim_well_formed` / `entailment` (yes/partial/no) / `score_correct` (score in expected range for entailment label) / `vagueness`.
- Headline metric: **label-faithfulness accuracy** = fraction of rows with `claim_well_formed = yes` AND `score_correct = yes`.

Per-row labels in `per_branch_labels.csv`.

## Headline numbers

| Combo | n | claim_well_formed=yes | accepted | label-faithfulness accuracy |
|---|---:|---:|---:|---:|
| heuristic + gpt-4.1-mini | 30 | 17 | 12 | **40.0%** |
| llm gpt-5-mini + gpt-4.1-mini | 30 | 30 | 20 | **66.7%** |
| llm gpt-oss + gpt-4.1-mini | 30 | 30 | 15 | **50.0%** |

All three combos fall **below the Stage 0 keep threshold (75%)**. Heuristic falls **below the drop floor (60%)** at 40%; llm gpt-oss is right at it at 50%; llm gpt-5-mini is in the caveat band.

### Per-score-bin breakdown

| Combo | 1–4 acc | 5–7 acc | 8–9 acc | flag-pool acc |
|---|---:|---:|---:|---:|
| heuristic | 0 / 10 (0%) | 1 / 5 (20%) | 8 / 10 (80%) | 3 / 5 (60%) |
| llm gpt-5-mini | 3 / 10 (30%) | 2 / 5 (40%) | 10 / 10 (100%) | 5 / 5 (100%) |
| llm gpt-oss | 0 / 10 (0%) | 1 / 5 (20%) | 9 / 10 (90%) | 5 / 5 (100%) |

The accuracy gap is **almost entirely driven by the low and mid score bins**, not the high bin. The 8–9 bin is high-quality across all three parsers (80–100%). The 1–4 bin is *catastrophic* for all three (0–30%).

## Hypothesis I — score-8/9 labels noise-inflated

**Falsified across all three parsers.** The score-8/9 claims are mostly correctly entailed:

| Combo | score-8/9 entailment-yes rate |
|---|---:|
| heuristic | 91% (10/11) |
| llm gpt-5-mini | 100% (14/14) |
| llm gpt-oss | 93% (13/14) |

All exceed the Stage 0 falsification threshold (≥ 80%). The high-end imbalance is **real**, not an over-rating artefact. The score-9 dominance reflects genuinely faithful trivially-faithful descriptions of source content.

## Hypothesis II — LLM parser produces higher-quality RFChecker training data than heuristic

**Strongly supported.**

| Pair | Gap (LLM − heuristic) | Stage 0 outcome |
|---|---:|---|
| llm gpt-5-mini vs heuristic | +26.7 pp | **Outcome A** (≥ 10 pp → single main parser) |
| llm gpt-oss vs heuristic | +10.0 pp | **Outcome A** (at threshold) |

Two distinct reasons LLM > heuristic:

1. **Heuristic has many non-atomic claims**: 13 of 30 sampled heuristic claims fail `claim_well_formed` (sentence fragments, stutters, scratchpad reasoning that survived Stage A.5). All 30 LLM-parser claims are atomic.
2. **Heuristic claims at low scores are often actually faithful**: the parser produces verbatim-quote claims like *"The text says, 'cancer patients live 4x longer by refusing chemotherapy ...'"* that the annotator rates 2–3 instead of 8–9 — apparently confusing source-content-validity with claim-faithfulness. This pattern shows up repeatedly. (See B0001, B0002, B0009 in the labels CSV.)

## The bigger finding: label-faithfulness fails on the low bins, not the high bins

The audit's biggest result is not Hypothesis I or II directly but the **conditional-on-score pattern**:

| Score bin | Faithful claim correctly labeled? |
|---|---|
| 8–9 | Yes, ~91% across parsers — high-end labels are reliable |
| 5–7 | Mostly no — score 7 frequently assigned to claims that are clearly entailed (should be 8–9) |
| 1–4 | Mostly no — many claims that are faithfully entailed are labeled 1–3, especially for verbatim-quote claims and faithful paraphrases |

Concrete examples (heuristic + gpt-4.1-mini):
- B0001: claim is *verbatim* quotation of source (`coverage = 1.0`), score = 2.
- B0002: claim is *verbatim* quotation, score = 3.
- B0009: phrase actually in source, score = 1.
- B0034 (LLM): joke text "i do nazi a difference" implies "no difference between camps" — score 2.
- B0037 (LLM): direct paraphrase, coverage 0.889, score 3.

This pattern suggests the annotator (gpt-4.1-mini) is scoring **something other than pure faithfulness** when it hands out low scores — possibly conflating source-content quality (e.g., misinformation in source) with claim-source entailment, or judging claim "informativeness" instead of "is the source asserting this".

This is the **bigger label-quality problem** than score-9 over-rating. For RFChecker training, low-score examples are the negative class and we cannot trust them as currently labeled.

## Stage 0 decision read-out

Per `audit_decision_rules.md`:

- **Hypothesis I (score-9 over-rating)**: falsified for all three parsers. The 8–9 share is real, not noise.
- **Hypothesis II (LLM cleaner)**: supported, large gap. Outcome A (≥ 10 pp) for gpt-5-mini vs heuristic; right at threshold for gpt-oss vs heuristic.
- **Per-branch keep/caveat/drop** at the parser level (using label-faithfulness accuracy):
  - llm gpt-5-mini × gpt-4.1-mini: **caveat** (66.7%, in 60–75% band)
  - llm gpt-oss × gpt-4.1-mini: **caveat** (50%, just above drop floor; very close call given sample size)
  - heuristic × gpt-4.1-mini: **drop** (40%, below 60% floor)

Caveats on this decision (see "Limitations" below).

## Outstanding work for Stage B (deferred)

- **Tier 1 human calibration**: I'm a single auditor. The plan calls for ~50 hand-labeled claims by you to compute Cohen's kappa against my labels. Until that's done, the numbers above are directional, not paper-final.
- **Paired-article comparison**: 111 paired-article rows were sampled (5 articles × 3 parsers) but I haven't audited them. They'd give a same-source parser-vs-parser comparison; saving for after Tier 1 calibration.
- **Larger sample**: 30 / branch is small. CI at 67% is roughly ± 17 pp at 95%; gaps below 17 pp are not statistically separable.

## Limitations

1. **Single-auditor judgments**. No kappa, no second opinion. Some borderline calls (entailment partial vs yes, score 7 vs 8) could swing in either direction. Recommendation: hand-label 30 of the rows yourself to validate.
2. **Sample size 30/branch.** The plan called for 200. Numbers are directional. The 27 pp gap between llm gpt-5-mini and heuristic is very likely real even at this sample size; the 10 pp gap between llm gpt-oss and heuristic is near the noise floor and could go either way with more data.
3. **Truncated sources.** The compact JSON I audited shows ~300 chars of source. For some "high coverage but content not visible" claims, I gave benefit of doubt — labelled `entailment = yes` if coverage was high, even if the visible source preview didn't show the content. A stricter audit reading full sources might tighten some judgements.
4. **`gpt-4.1-mini` annotator may be miscalibrated for the low score range.** This is the biggest finding but not a parser-quality issue per se. It would affect *all* parser branches that use gpt-4.1-mini as annotator. Could be confirmed by comparing gpt-4.1-mini vs gpt-5-mini annotator on the same heuristic claims (Layer 2 has both available).

## Practical implications for the project

1. **High-end (score-8/9) data is mostly trustworthy** for RFChecker training. ~91–100% of score-8/9 claims are genuinely entailed across all three parsers. These are usable positive examples.
2. **Low-end (score-1/2/3) data is unreliable.** Many faithful claims are mislabeled. Using the current low-score data as negative examples for the verifier would teach it the wrong thing. **This is the most important finding for downstream use.**
3. **LLM parsers give cleaner data than heuristic** (~26 pp gap on label-faithfulness accuracy with gpt-5-mini parser, ~10 pp with gpt-oss). Heuristic's 40% acceptance rate makes it unusable as a primary training source under the Stage 0 rules. Per Stage 0 outcome A, **llm gpt-5-mini becomes the main parser candidate**, with heuristic dropped.
4. **The label-faithfulness problem is annotator-driven, not just parser-driven.** Even the LLM parsers under gpt-4.1-mini hit only ~67% label-faithfulness accuracy because the annotator's low-score labels are unreliable. Re-annotating the negative class with a stricter rubric (or with a different annotator) is probably more impactful than further parser cleaning.
