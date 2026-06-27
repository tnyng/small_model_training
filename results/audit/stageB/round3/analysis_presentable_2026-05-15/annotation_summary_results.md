# Stage B Round 3 — Results

## Scope and cleaning
- Full analysis uses **53 unique claims** and **160 total human labels** after merging the 5 overlap claims with earlier Yang/Yu labels.
- Applied cleaning: `ye s` -> `yes`, `?` -> blank, `n/a` -> blank, `partial` in `claim_well_formed` -> blank.
- Tobi has one missing row, which is kept as missing and ignored in judged-field metrics.

## Round-3 Annotator Coverage (5 annotators only)
| annotator | rows | cwf_yes | cwf_no | cwf_blank | ent_yes | ent_partial | ent_no | ent_blank | sc_yes | sc_partial | sc_no | sc_blank | notes_rows |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Alexander | 30 | 24 | 2 | 4 | 14 | 11 | 5 | 0 | 18 | 6 | 5 | 1 | 20 |
| Jingwei | 30 | 20 | 8 | 2 | 19 | 3 | 8 | 0 | 22 | 0 | 8 | 0 | 15 |
| Tobi | 30 | 24 | 5 | 1 | 14 | 9 | 1 | 6 | 18 | 1 | 5 | 6 | 7 |
| Yang | 30 | 29 | 1 | 0 | 21 | 1 | 8 | 0 | 27 | 0 | 3 | 0 | 0 |
| Yu | 30 | 28 | 2 | 0 | 22 | 5 | 1 | 2 | 21 | 0 | 7 | 2 | 0 |

## Agreement Summary (full merged 53-claim set)
| field | items_with_2plus_labels | complete_items | mean_pairwise_agreement | fleiss_kappa |
|---|---|---|---|---|
| claim_well_formed | 53 | 46 | 0.877 | 0.373 |
| entailment | 51 | 47 | 0.627 | 0.308 |
| score_correct | 51 | 46 | 0.706 | 0.359 |
| pass_strict (cwd=yes&sc=yes) | 53 | 46 | 0.714 | 0.379 |
| pass_lenient (cwd=yes&sc=y/p) | 53 | 46 | 0.745 | 0.459 |

## Main Audit Signal by Score Bin
| score_bin | n_judged | strict_pass_rate |
|---|---|---|
| 1-4 | 57 | 0.316 |
| 5-7 | 50 | 0.76 |
| 8-9 | 46 | 0.935 |

## Branch Comparison (strict pass)
| branch | n_judged | strict_pass_rate |
|---|---|---|
| heuristic_ann_gpt41mini | 45 | 0.644 |
| llm_gpt5mini_ann_gpt41mini | 46 | 0.761 |
| llm_gptoss_ann_gpt41mini | 48 | 0.667 |

## Majority-Label Summary Across Claims
| field | majority_label | count |
|---|---|---|
| claim_well_formed | no | 4 |
| claim_well_formed | yes | 49 |
| entailment | TIE | 7 |
| entailment | no | 6 |
| entailment | partial | 8 |
| entailment | yes | 32 |
| score_correct | TIE | 3 |
| score_correct | no | 7 |
| score_correct | partial | 1 |
| score_correct | yes | 42 |
| pass_strict | TIE | 2 |
| pass_strict | no | 15 |
| pass_strict | yes | 36 |
| pass_lenient | TIE | 3 |
| pass_lenient | no | 12 |
| pass_lenient | yes | 38 |

## Annotator notes

| theme | count | example_1 | example_2 |
|---|---|---|---|
| Parser / formatting artifacts | 5 | Alexander: difficult one; the claim makes it seem like that's all the text is doing, but it is strictly true that the text contains this claim. Should be a 6 or 7 | Alexander: is heursitic working correctly? "the text" seems appended incorrectly to the start here |
| Borderline score-range judgment | 10 | Alexander: should be 9? | Alexander: perhaps an example of input text that should have been filtered out? 7 is much too high here |
| Claim vagueness / complexity | 4 | Alexander: brings up additional evidence in the parsing ? I think should be in the 4ish range, it's a weird claim though | Jingwei: Hard to say it is a claim |
| Source-language / source-reading issues | 1 | Tobi: the text is partially in an Indian language |  |


## Takeaways
- The main ranking is unchanged: `llm_gpt5mini_ann_gpt41mini` remains strongest, `llm_gptoss_ann_gpt41mini` next, `heuristic_ann_gpt41mini` weakest.
- The most important data-quality result remains unchanged: score **8–9** rows are strong positives, while score **1–4** rows are weak natural negatives.
- Human agreement is meaningful but only moderate; this supports using human annotation as audit evidence rather than treating any single label source as absolute ground truth.
- For downstream verifier / reward-style training, the data still point toward: trust high-score positives, avoid raw low-score negatives without extra processing or synthesis.