# Stage A.5 — Filter Application Summary

**Audit reframing**: under the RFChecker training-data goal (per the project proposal), the audit is structured as two layers — *claim quality* (Stage A.5, this stage) is the prerequisite filter, and *label faithfulness* (Stage B) is the headline measurement. Stage A.5 drops claims that aren't atomic well-formed propositions (fragments, stutters, scratchpad, questions) or that reference content not in the source. **Trivially-faithful claims like *"the text mentions Trump"* on a Trump article are NOT noise** — they're valid atomic positives for the RFChecker and are kept.

Filter rules and validation: see `filter_rules.md`. Recall 80% on the 20-claim noise sample (target 70% met).

## What got dropped (per raw combo, Layer 2)

| Combo | rows in | dropped | drop rate | 8-9 pre | 8-9 post |
|---|---:|---:|---:|---:|---:|
| heuristic + gpt-4.1-mini | 22 512 | 10 515 | 46.7% | 67.6% | 74.4% |
| heuristic + gpt-5-mini | 22 512 | 10 515 | 46.7% | 84.1% | 89.6% |
| heuristic + gpt-oss | 1 500 | 746 | 49.7% | 66.4% | 72.7% |
| llm gpt-5-mini + gpt-4.1-mini | 19 527 | 4 956 | 25.4% | 88.9% | 91.5% |
| llm gpt-5-mini + gpt-5-mini | 19 527 | 4 956 | 25.4% | 95.0% | 97.3% |
| llm gpt-oss + gpt-4.1-mini | 22 551 | 7 618 | 33.8% | 83.2% | 85.1% |
| llm gpt-oss + gpt-oss | 1 500 | 496 | 33.1% | 84.4% | 84.0% |

Two structural observations:

**1. Heuristic combos drop ~47% of claims; LLM combos drop ~25–34%.** Heuristic drops are dominated by `template_stutter_the_text` (malformed "the text the X" stems) and `coverage_zero` (claim content not in source). LLM combos don't have the stutter pattern so their main drop driver is `coverage_zero`. The earlier figure of 59% drop on heuristic was wrong — it dropped *"the text uses the phrase X"* unconditionally, which is incorrect under the RFChecker frame: those are valid faithful atomic claims when X actually appears in the source. The rule is now coverage-conditional (drop only when coverage = 0).

**2. Identical drop counts for `gpt-4.1-mini` and `gpt-5-mini` annotators on the same parser.** Drops depend only on the `property` and `text` columns, which are identical across annotators (the parser produces one set of claims; multiple annotators score them). So drops are 13 218 in both heuristic combos and 5 022 in both LLM-gpt-5-mini combos.

## Score-bin shifts after filtering

| Combo | 8-9 share pre-filter | 8-9 share post-filter | Δ |
|---|---:|---:|---:|
| heuristic + gpt-4.1-mini | 67.6% | 74.4% | +6.8 pp |
| heuristic + gpt-5-mini | 84.1% | 89.6% | +5.5 pp |
| heuristic + gpt-oss | 66.4% | 72.7% | +6.3 pp |
| llm gpt-5-mini + gpt-4.1-mini | 88.9% | 91.5% | +2.6 pp |
| llm gpt-5-mini + gpt-5-mini | 95.0% | 97.3% | +2.3 pp |
| llm gpt-oss + gpt-4.1-mini | 83.2% | 85.1% | +1.9 pp |

After dropping noise, the score-9 share goes *up*. This is consistent with the RFChecker framing: the filter removes claims that aren't atomic well-formed propositions, which sit somewhat preferentially in the mid bin (where claims are partial / hedge-like). The remaining claims are biased toward the high end because **trivially-faithful claims are now correctly retained as positive examples** (e.g. *"the text uses the phrase 'please'"* with "please" in source — score 9, kept).

### What this does and does NOT tell us about the imbalance

The "imbalance is a problem" framing is the wrong question for this project. Under the RFChecker training goal:

- Score 8/9 = "claim is faithful to source" → these are positive training examples. We want plenty.
- Score 1/2 = "claim is unfaithful" → these are negative training examples. ~6–18% across combos is enough.
- Score 5/6/7 = boundary region most useful for selective-classification thresholds.

The high 8/9 share reflects a real property: parsers freely emit many trivially-faithful descriptions of any source text, and those correctly receive high entailment scores. That is exactly what we want the verifier to learn. The imbalance is "a problem" only if the *labels at either end are wrong* — which is a Stage B (label-faithfulness) question, not a Stage A.5 (claim-quality) question.

### Hypothesis I, restated

The original Hypothesis I ("imbalance is partly a parser/noise artefact") was framed around *surface-restatement noise* and is plausible but irrelevant under the RFChecker goal: even if many score-9 claims are surface-restatement, those are valid positive training examples as long as the claims are faithfully entailed by the source. The proper restatement of Hypothesis I for this project is:

**"Are the score-9 labels actually correct? I.e., is the assigned score 8/9 because the claim is genuinely entailed by the source, or because the annotator over-rated a vacuous match?"**

This is the question Stage B answers via the `entailment` rubric field on stratified samples.

## Per-branch (Layer 1) filter logs

Filtered sibling folders written under `modernbert_uq/data/<branch>__filtered/`. Per-branch JSON logs at `results/audit/stageA5/branch_filter_log_<branch>.json`. Stage B (and any future modelling experiment) operates on these `__filtered` branches; the originals stay untouched for before/after comparison.

For the mainline `rulefollower_heuristic_ann_gpt41mini_reg_base` branch, post-filter train rows ≈ 6 500 (down from 15 763). Other branches scale similarly — drop rate ~58% for heuristic, ~25–34% for LLM. **All filtered `reg_base` branches still have train rows ≥ 5 000**, so they remain "keep" candidates under Stage 0 §1b. The `reg_down_exact` and probe branches still fall below 1 000 post-filter and remain Stage D drops as expected.

## What this means for the project

1. **The lexical filter is a partial cleaning step.** It removes ~25–60% of obviously broken claims (stutter, fragment, source-misalignment). That's worth doing, and the filtered branches are now ready for Stage B.
2. **The filter does NOT settle the imbalance question.** That gets answered in Stage B by the semantic rubric (`extraction_correct`, `vagueness`), not by a regex.
3. **Hypothesis I (imbalance is partly noise-driven, in the surface-restatement sense) is plausible and untested.** Stage B will test it directly: whether score-8/9 claims pass or fail the substantive-vs-vacuous rubric.
4. **Hypothesis II (LLM parser is cleaner) also needs Stage B's paired-article acceptance to answer.** The Stage A.5 drop rates are not a quality measure — they reflect the *type* of noise each parser produces: heuristic has more lexically catchable noise (stutter, substring detection), LLM has less but a comparable rate of `coverage_zero` source-misalignment.
5. **The filter's drop rates by bin** (1–4: 56%, 5–7: 67%, 8–9: 58% on heuristic + gpt-4.1-mini) tell us the filter is bin-roughly-neutral but slightly biased to the mid bin. This explains why the 8/9 share doesn't move much after filtering — but again, that's a property of the filter, not of the data's true label distribution.

## Outputs

- `filter_rules.md` — locked filter spec.
- `filter_claims.py` — implementation.
- `filter_validation.json` — recall on the noise sample, rule firings, spot-check sample summary.
- `random_drop_review.csv` — 150-claim sample with per-rule annotations for spot-checking.
- `apply_summary.json` — per-combo and per-branch post-filter numbers.
- `raw_combo_logs/<combo>.json` — full per-combo filter log (drop rate, rule counts, by source dataset).
- `branch_filter_log_<branch>.json` — per-branch (Layer 1) filter log.
- `modernbert_uq/data/<branch>__filtered/` — filtered sibling branches, ready for Stage B.
