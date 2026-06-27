# Stage B Round 3 — Presentation Note (Cleaned Full-53 Analysis)

## Scope
- Merge round3 with the earlier two human-labeled overlap files so the 5 overlap claims have their intended 3-human-label view.
- Clean only agreed anomalies: `ye s` -> `yes`, `?` -> blank, `n/a` -> blank, `partial` in `claim_well_formed` -> blank.
- Keep `downstream_filled_after_no` rows as valid annotation style, not errors.
- Keep Tobi's one blank row as missing and ignore it in judged-field analyses.

## Dataset Size
- Unique claims: **53**
- Total annotation rows: **160**
- Label-count histogram across claims: **52 claims with 3 labels, 1 claim with 4 labels**

## Agreement Summary
| field | items_with_2plus_labels | complete_items | mean_pairwise_observed_agreement | fleiss_kappa_complete_items |
|---|---|---|---|---|
| claim_well_formed | 53 | 46 | 0.877 | 0.373 |
| entailment | 51 | 47 | 0.627 | 0.308 |
| score_correct | 51 | 46 | 0.706 | 0.359 |
| pass_strict | 53 | 46 | 0.714 | 0.379 |
| pass_lenient | 53 | 46 | 0.745 | 0.459 |

## Per-Annotator Summary
| annotator | rows | blank_rows | notes_rows | cwf_yes | cwf_no | cwf_blank | ent_yes | ent_partial | ent_no | ent_blank | sc_yes | sc_partial | sc_no | sc_blank |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A1_prior | 5 | 0 | 0 | 5 | 0 | 0 | 3 | 2 | 0 | 0 | 1 | 3 | 1 | 0 |
| A2_prior | 5 | 0 | 0 | 5 | 0 | 0 | 4 | 1 | 0 | 0 | 2 | 0 | 3 | 0 |
| Alexander | 30 | 0 | 20 | 24 | 2 | 4 | 14 | 11 | 5 | 0 | 18 | 6 | 5 | 1 |
| Jingwei | 30 | 0 | 15 | 20 | 8 | 2 | 19 | 3 | 8 | 0 | 22 | 0 | 8 | 0 |
| Tobi | 30 | 1 | 7 | 24 | 5 | 1 | 14 | 9 | 1 | 6 | 18 | 1 | 5 | 6 |
| Yang | 30 | 0 | 0 | 29 | 1 | 0 | 21 | 1 | 8 | 0 | 27 | 0 | 3 | 0 |
| Yu | 30 | 0 | 0 | 28 | 2 | 0 | 22 | 5 | 1 | 2 | 21 | 0 | 7 | 2 |

## Main Audit Signal by Score Bin
| bin | strict_pass_n | strict_pass_rate |
|---|---|---|
| 1-4 | 57 | 0.316 |
| 5-7 | 50 | 0.760 |
| 8-9 | 46 | 0.935 |

## Branch Comparison (Strict Pass)
| branch | strict_pass_n | strict_pass_rate |
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

## Overlap Claims (Merged Human Labels)
| row_id | annotator | claim_well_formed | entailment | score_correct | notes |
|---|---|---|---|---|---|
| B0065 | A1_prior | yes | partial | partial |  |
| B0004 | A1_prior | yes | partial | partial |  |
| B0015 | A1_prior | yes | yes | yes |  |
| B0033 | A1_prior | yes | yes | partial |  |
| B0032 | A1_prior | yes | yes | no |  |
| B0033 | A2_prior | yes | partial | yes |  |
| B0004 | A2_prior | yes | yes | no |  |
| B0032 | A2_prior | yes | yes | no |  |
| B0065 | A2_prior | yes | yes | no |  |
| B0015 | A2_prior | yes | yes | yes |  |
| B0033 | Alexander | yes | yes | partial | difficult one; the claim makes it seem like that's all the text is doing, but it is strictly true that the text contains this claim. Should be a 6 or 7 |
| B0004 | Alexander |  | partial | no | should be a higher score |
| B0065 | Jingwei | yes | yes | no | Interesting |
| B0015 | Tobi | no |  |  |  |
| B0032 | Tobi | yes | yes | no | are we good with more complicated claims? |

## Cleaning Log
| source_file | sheet_row | row_id | field | raw | cleaned |
|---|---|---|---|---|---|
| Annotation_Alexander.csv | 20 | rumoureval2019__fc4614f6fdf337b0__claim_14230 | claim_well_formed | partial |  |
| Annotation_Alexander.csv | 21 | tweets_news__75d9dd2099921ee1__claim_18311 | claim_well_formed | partial |  |
| Annotation_Alexander.csv | 21 | tweets_news__75d9dd2099921ee1__claim_18311 | score_correct | n/a |  |
| Annotation_Alexander.csv | 27 | B0004 | claim_well_formed | partial |  |
| Annotation_Alexander.csv | 29 | misinfo__cafc1c58d509a29d__claim_7968 | claim_well_formed | partial |  |
| Annotation_Jingwei.csv | 4 | tweets_news__f276eaab226cf2ab__claim_18086 | claim_well_formed | ? |  |
| Annotation_Jingwei.csv | 18 | article_bias__64e61a5af04ced88__claim_1213 | claim_well_formed | ? |  |
| Annotation_Yang.csv | 21 | tweets_news__75d9dd2099921ee1__claim_18311 | claim_well_formed | ye s | yes |

## Takeaways
- The cleaned full-53 analysis keeps the same ranking: `llm_gpt5mini_ann_gpt41mini` strongest, `llm_gptoss_ann_gpt41mini` next, `heuristic_ann_gpt41mini` weakest.
- The strongest data-quality result remains score-bin driven: 8–9 rows are very strong positives, 1–4 rows are weak natural negatives.
- Human agreement is meaningful but not so high that we should treat the rubric as fully settled; disagreement remains concentrated in malformed/borderline/low-score cases.
- For downstream verifier / reward-style training, the result still points toward: trust high-score positives, avoid using raw low-score rows as negatives without extra processing.