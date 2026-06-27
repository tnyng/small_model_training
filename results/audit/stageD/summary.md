# Stage D — Branch Triage

Applies Stage 0 §1b thresholds and Stage C parser decision to all 34 modernbert experiment branches. Output: `branch_triage.csv` with one row per branch.

## Headline

| Decision | Count |
|---|---:|
| keep | 3 |
| caveat | 9 |
| drop | 22 |

Of the 34 branches, **only 3 land in the "keep" bucket** — and they're the 3 unaudited llm gpt-5-mini × gpt-5-mini reg/threshold branches that escape the LF-accuracy check by virtue of not having Stage B numbers (more on that caveat below).

## Per-decision summary

### Keeps (3 branches)

All three are `llm_gpt5mini × gpt-5-mini` annotator + the threshold/regression base variants:
- `rulefollower_llm_gpt5mini_ann_gpt5mini_reg_base` (train post-filter 10 148)
- `rulefollower_llm_gpt5mini_ann_gpt5mini_ge8` (10 148)
- `rulefollower_llm_gpt5mini_ann_gpt5mini_gt8` (10 148)

These pass all numerical thresholds (train ≥ 5 000, no leakage, no rebalancing pathology) and have **no Stage B label-faithfulness audit** (the gpt-5-mini annotator wasn't sampled in Stage B). They're "kept" by the absence of evidence against them, not by positive evidence. Treat with caution.

### Caveats (9 branches)

All `llm` parser branches that fail one threshold without triggering a drop. Mostly:
- Three `llm_gpt5mini × gpt-4.1-mini` reg_base / ge8 / gt8 — caveat for **LF = 73%** (in 60–75% band).
- One `llm_gpt5mini × gpt-4.1-mini reg_down_bins` — caveat for both LF and train ∈ [1 000, 5 000).
- Two `llm_gptoss × gpt-4.1-mini` ge8 / gt8 — caveat for **LF = 60%** (right at floor).
- Three `heuristic × gpt-5-mini` reg_base / ge8 / gt8 — caveat from parser-role demotion (Stage C), no Stage B numbers, but pass size/leakage.

### Drops (22 branches)

Three concentrated reasons:

**Train size < 1 000** (the broken slices, robust drop regardless of view):
- `llm_gpt5mini × gpt-4.1-mini reg_down_exact` (train 18, post-filter 13)
- `llm_gpt5mini × gpt-5-mini reg_down_exact` (216 → 124)
- `llm_gpt5mini × gpt-5-mini reg_down_bins` (600 → 361)
- `heuristic × gpt-oss ge8 / gt8` (1 013 → 521)
- `llm_gptoss × gpt-oss ge8 / gt8` (1 050 → 724)

**Rebalancing pathology** (single source claim duplicated > 5×, post-rebalance dup > 5%):
- All `reg_over_exact` slices (max-id rep 124× to **5 566×**).
- Some `reg_over_bins` slices (rep 14× to 84×).

**Label-faithfulness < 60%** (heuristic + gpt-4.1-mini family, all 7 variants):
- All `rulefollower_heuristic_ann_gpt41mini_*` branches drop because their parent combo's combined-lenient LF accuracy is **50%**, below the 60% drop floor. *(See the sensitivity note below — under user-only-strict LF, this combo sits at 60% and would shift to caveat.)*

## Sensitivity to the LF accuracy view

This triage uses **combined-lenient** Stage B accuracies (the largest-sample point estimate):

| Combo | Combined-lenient LF | User-only-strict LF |
|---|---:|---:|
| heuristic + gpt-4.1-mini | 50% | 60% |
| llm gpt-5-mini + gpt-4.1-mini | 73% | 56% |
| llm gpt-oss + gpt-4.1-mini | 60% | 70% |

If we used user-only-strict instead, the LF-driven decisions would change:

- `heuristic × gpt-4.1-mini` branches → **caveat** (60% is right at floor, not below).
- `llm_gpt5mini × gpt-4.1-mini` branches → **drop** (56% < 60%) — that would invalidate Stage C's choice of gpt-5-mini as main parser.
- `llm_gptoss × gpt-4.1-mini` branches → **caveat** (70% in 60–75% band).

These two views disagree because we have only 10 rows per combo for user-verified data and 30 rows per combo for combined data. The 95% CI on either is wider than the gap between them.

**The conservative / honest read**: at this sample size, **all three audited combos land in the "caveat" band** (60–75%) under one view or another. None is robustly above 75% (keep), none robustly below 60% (drop). The mechanical Stage 0 read just picks one corner of the uncertainty.

## What survives if you take the conservative read

If we treat all three audited combos as "caveat" regardless of view, the triage becomes:
- **Keep**: 0 branches with full evidence (the 3 "keeps" are unaudited, so they're keep-by-default).
- **Caveat**: 9 + ~7 (the heuristic gpt-4.1-mini family) = ~16 branches.
- **Drop**: ~18 branches (those failing size or rebalancing rules — these drops are robust regardless of LF view).

The 18 robust drops are: every `_reg_down_exact` slice (train ≤ 1 071), most `_reg_over_*` slices (id-repetition pathology), and all `_gpt-oss_ann_*` 1 500-row probes. These survive any sensitivity test.

## Recommended use

For modelling experiments going forward (and for paper-track reportable claims):

1. **Primary training source**: `llm_gpt5mini × gpt-4.1-mini reg_base` (caveat — verified Stage B but at borderline LF accuracy).
2. **Cross-annotator robustness check**: `llm_gpt5mini × gpt-5-mini reg_base` (kept, but unaudited at Stage B; treat as same parser, different annotator point).
3. **Cross-parser robustness check**: `llm_gptoss × gpt-4.1-mini ge8 / gt8` (caveat; threshold tasks only since no reg_base for this combo).
4. **Robustness/appendix only**: heuristic-parser branches at the `_gpt5mini` annotator, where size and leakage pass.

For RFChecker training corpus assembly:
- Drop every `_reg_down_exact` and `_reg_over_*` branch (mechanical degeneracy).
- Drop every gpt-oss-annotator probe combo (too small for headline use).
- Keep the `reg_base` family from `llm_gpt5mini × gpt-4.1-mini` and `llm_gpt5mini × gpt-5-mini` as the candidate corpus.

## Outstanding work

- Hand-label the second-annotator pack (v2) when ready — re-running `merge_v2_labels.py` and `recompute_with_human_labels.py` will tighten the Stage B numbers and may shift Stage D decisions for the borderline branches.
- Stage F (calibration) on the kept/caveat branches will tell us whether these survive temperature-scaling / MC-dropout reliability checks.

## Outputs

- `branch_triage.csv` — full per-branch decisions with reasons.
- `run_stage_d.py` — script to regenerate.
