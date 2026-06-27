# Round 3 Audit Summary: 3 LLM Auditors vs 5 Humans

## Scope

This note summarizes the round-3 audit comparison between:

- 5 human annotators: `Yang`, `Yu`, `Alexander`, `Jingwei`, `Tobi`
- 3 LLM auditors: `Opus 4`, `Sonnet 4.6`, `Codex / GPT-5.4`

Important correction:

- `A1_prior` and `A2_prior` in the cleaned merge are not extra people.
- They are the earlier overlap labels from `Yang` and `Yu`, carried forward into round 3.
- The correct framing is therefore **3 models vs 5 humans**, not 3 models vs 7 humans.

## Human Annotation Design

From the stage-B files:

- Round 1: one human label set in `human_labels_blank_filled.csv`
- Round 2: one human label set in `sample_label_yu.csv`
- Round 2 overlap: 5 overlap rows recorded in `user_labels_v2_overlap.csv`
- Round 3: 5-human design documented in `round3/build_round3.py`
  - 48 new claims
  - 5 overlap claims from earlier rounds
  - intended 3 human labels per row total
  - the 5 overlap rows already had Yang+Yu labels, so they received one additional human label in round 3

This means the human reference set is a **3-label human panel per row**, assembled from the 5 named annotators above.

## Why Pairwise Agreement Is The Primary Metric

For this audit design, the right primary comparison is **pairwise**:

- human vs human:
  compare available human labels pairwise and average
- model vs human:
  compare the model label to each available human label on the same row and average

This is already how the human-only round-3 analysis is summarized in `agreement_summary_full53.csv`, through:

- `mean_pairwise_observed_agreement`

Fleiss' kappa is still useful, but should be treated as a **secondary panel-consistency metric** rather than the main score.

## Human-Only Agreement

Source: `analysis_cleaned2_2026-05-15/agreement_summary_full53.csv`

| Field | Mean Pairwise Agreement | Fleiss' Kappa |
|---|---:|---:|
| `claim_well_formed` | 0.877 | 0.373 |
| `entailment` | 0.627 | 0.308 |
| `score_correct` | 0.706 | 0.359 |
| `pass_strict` | 0.714 | 0.379 |
| `pass_lenient` | 0.745 | 0.459 |

Interpretation:

- Humans agree strongly on `claim_well_formed`.
- `entailment` is the hardest semantic field.
- `score_correct` and pass-style fields are easier than entailment, but still not trivial.

## Model vs 5 Humans: Pairwise Agreement

Source: `pairwise_vs_5human_labels.csv`

| Field | Codex / GPT-5.4 | Opus 4 | Sonnet 4.6 |
|---|---:|---:|---:|
| `claim_well_formed` | **0.908** | 0.850 | 0.837 |
| `entailment` | 0.664 | **0.684** | 0.493 |
| `score_correct` | 0.603 | 0.576 | **0.636** |
| `pass_strict` | 0.747 | **0.753** | 0.719 |
| `pass_lenient` | 0.644 | 0.630 | **0.678** |

Interpretation:

- `Codex / GPT-5.4` is best on `claim_well_formed`.
- `Opus 4` is best on `entailment`, the most important semantic field.
- `Sonnet 4.6` is clearly weakest on entailment, even though it looks somewhat better on downstream `score_correct` and `pass_lenient`.

## Model vs 5 Humans: Mean Cohen's Kappa

Method:

- compute Cohen's kappa separately for each `(model, human)` pair
- average across the 5 humans

Source: `model_vs_humans_kappa_summary.csv`

| Field | Codex / GPT-5.4 | Opus 4 | Sonnet 4.6 |
|---|---:|---:|---:|
| `claim_well_formed` | **0.539** | 0.528 | 0.406 |
| `entailment` | 0.389 | **0.438** | 0.268 |
| `score_correct` | **0.250** | 0.192 | 0.100 |
| `pass_strict` | 0.531 | **0.554** | 0.473 |
| `pass_lenient` | 0.344 | 0.347 | **0.370** |

Interpretation:

- Kappa is stricter than raw agreement because it discounts chance agreement and label imbalance.
- The same pattern remains:
  - Codex is strongest on `claim_well_formed`
  - Opus is strongest on `entailment`
  - Sonnet is weakest on entailment

## Fleiss' Kappa: Human Panel vs Human+Model Panel

Method:

- use rows with exactly 3 human labels for a field
- compute Fleiss' kappa for the 3-human panel
- add one model label as a 4th rater
- recompute Fleiss' kappa

Source: `fleiss_kappa_humans_vs_humansplusmodel.csv`

| Field | Humans Only | + Codex | + Opus 4 | + Sonnet 4.6 |
|---|---:|---:|---:|---:|
| `claim_well_formed` | 0.394 | 0.479 | **0.519** | 0.448 |
| `entailment` | 0.265 | 0.295 | **0.323** | 0.236 |
| `score_correct` | **0.358** | 0.286 | 0.241 | 0.261 |
| `pass_strict` | **0.733** | 0.620 | 0.646 | 0.603 |
| `pass_lenient` | **0.556** | 0.400 | 0.418 | 0.449 |

Interpretation:

- On the core semantic fields (`claim_well_formed`, `entailment`), **Opus 4** behaves like the best extra panel member.
- On `score_correct` and the pass fields, adding any model lowers panel consistency relative to the humans-only panel.
- This is a sign that LLMs are more helpful as semantic auditors than as direct replacements for the full downstream scoring judgment.

## Comparison To Human-Only Results

### Pairwise view

- Human-only pairwise `entailment`: 0.627
- Opus-vs-human pairwise `entailment`: 0.684
- Codex-vs-human pairwise `entailment`: 0.664
- Sonnet-vs-human pairwise `entailment`: 0.493

This suggests:

- `Opus 4` and `Codex / GPT-5.4` are in the human-like range on entailment
- `Sonnet 4.6` is meaningfully worse on the hardest field

### Fleiss / panel-consistency view

- Humans-only panel remains strongest on `score_correct`, `pass_strict`, and `pass_lenient`
- Adding a model helps most on `claim_well_formed` and `entailment`, especially for `Opus 4`

## What This Means For Step-3 Accuracy Auditing

The audit goal is not to rank LLMs in general. The goal is to decide whether the **step-3 accuracy-score labels** are trustworthy enough for training data construction.

The results support the following interpretation:

1. The audit labels are **useful but noisy**, especially on the semantic boundary cases.
2. The most important distinction remains:
   - high-score rows are relatively trustworthy
   - low-score rows are much noisier
3. Among LLM auditors:
   - `Codex / GPT-5.4` is the strongest overall structural auditor
   - `Opus 4` is the strongest entailment auditor
   - `Sonnet 4.6` is weaker, mainly because of entailment

## Recommendation

For reporting:

1. **Primary metric**: pairwise agreement
2. **Secondary metric**: mean Cohen's kappa
3. **Panel consistency diagnostic**: Fleiss' kappa on humans-only vs humans+model

For practical audit use:

- keep humans as the anchor
- treat `Opus 4` and `Codex / GPT-5.4` as useful auxiliary auditors
- treat `Sonnet 4.6` as weaker evidence

If one model must be chosen:

- choose **Codex / GPT-5.4** for overall closeness to human audit behavior

If a field-specific hybrid is allowed:

- use **Codex** for `claim_well_formed`
- use **Opus 4** for `entailment`
- derive downstream pass decisions from those semantic labels, rather than trusting a single model's `score_correct` field directly
