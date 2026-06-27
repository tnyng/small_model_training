# Stage B Round 3 — Cleaned + Merged Analysis

## Cleaning rules applied
- `ye s` -> `yes`
- `?` -> blank
- `n/a` -> blank
- `downstream_filled_after_no` was **not** treated as an error; those rows were kept as annotated
- Tobi's one fully blank row was kept as missing and ignored in judged-field analyses

## Final merged dataset size
- Unique claims: **53**
- Total annotation rows after merging overlap with earlier A1/A2 files: **160**
- Overlap claims merged from earlier rounds: **5**
- Label-count histogram across claims: {3: 52, 4: 1}

## Full-53 agreement summary
- `claim_well_formed`: mean pairwise agreement **0.827**, Fleiss kappa **0.284**, complete items **50**
- `entailment`: mean pairwise agreement **0.627**, Fleiss kappa **0.308**, complete items **47**
- `score_correct`: mean pairwise agreement **0.706**, Fleiss kappa **0.359**, complete items **46**
- `pass_strict`: mean pairwise agreement **0.714**, Fleiss kappa **0.379**, complete items **46**
- `pass_lenient`: mean pairwise agreement **0.745**, Fleiss kappa **0.459**, complete items **46**

## Main audit signal
- Bin 1-4: strict pass **31.6%** over **57** judged labels
- Bin 5-7: strict pass **76.0%** over **50** judged labels
- Bin 8-9: strict pass **93.5%** over **46** judged labels

## Branch comparison (strict pass)
- heuristic_ann_gpt41mini: **64.4%** (45 judged labels)
- llm_gpt5mini_ann_gpt41mini: **76.1%** (46 judged labels)
- llm_gptoss_ann_gpt41mini: **66.7%** (48 judged labels)

## Interpretation
- The branch ranking is unchanged after cleaning and overlap merge: `llm_gpt5mini_ann_gpt41mini` remains strongest, `llm_gptoss_ann_gpt41mini` next, `heuristic_ann_gpt41mini` weakest.
- The most important substantive result also survives: low-score rows are much weaker than 8–9 rows, so natural negatives remain questionable for verifier-style training.
- The overlap merge gives the 5 disputed claims their intended 3-human-label view, which is the right basis for final overlap inspection.