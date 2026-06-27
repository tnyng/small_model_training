# Stage B Round 3 — Analysis Summary

## What round 3 was designed to deliver
- Five-person human annotation to strengthen the data-quality audit, not just one-vs-Claude calibration.
- Primary value: better evidence about which claim/parser branches are trustworthy, especially whether low-score rows are truly useful negatives or mostly label/pathology artifacts.
- Secondary value: adjudication signal on disputed cases from earlier rounds via the 5 overlap rows.

## Pack structure confirmed from files
- Unique claims in round3 pack: **53**
- New claims: **48**
- Overlap claims from earlier rounds: **5**
- Within round3 files: 47 claims have 3 labels, 1 claim has 4 labels, 5 overlap claims have 1 new label each
- Across the full intended design, the 5 overlap claims reach 3 total human labels only when combined with earlier Stage B files.

## Completion / data hygiene
- Alexander: 30 assigned, 0 fully blank, 20 rows with notes, 0 rows where `claim_well_formed=no` and downstream fields were left blank
- Jingwei: 30 assigned, 0 fully blank, 15 rows with notes, 0 rows where `claim_well_formed=no` and downstream fields were left blank
- Tobi: 30 assigned, 1 fully blank, 7 rows with notes, 5 rows where `claim_well_formed=no` and downstream fields were left blank
- Yang: 30 assigned, 0 fully blank, 0 rows with notes, 0 rows where `claim_well_formed=no` and downstream fields were left blank
- Yu: 30 assigned, 0 fully blank, 0 rows with notes, 2 rows where `claim_well_formed=no` and downstream fields were left blank
- File-format issue found: `Annotation_Tobi.csv` uses CP1252 encoding and `;` delimiter instead of UTF-8 comma CSV, but it is still parseable and structurally consistent.

## What the human annotations are most useful for
1. Checking whether malformed / awkward claims are concentrated in heuristic outputs.
2. Checking whether low-score rows are really valid negatives for verifier-style training.
3. Measuring how much humans converge on the same label versus how much the rubric itself is still ambiguous.
4. Distinguishing parser problems from rubric/annotation problems.

## What the current round indicates
- Across all 150 annotation rows, `claim_well_formed=no` appears **18** times; these are valuable because they directly identify unusable claims rather than merely low-score claims.
- There are **8** blank `score_correct` cells; inspection shows these are mostly not random missingness but come from the annotation style 'if claim_well_formed=no, do not judge entailment/score_correct'.
- This means raw agreement on `score_correct` must be computed on judged rows only; for an audit-value metric, malformed claims should still count as failures in a derived pass/fail view.
- Agreement on `claim_well_formed`: mean pairwise observed agreement = **0.802** across items with at least 2 judgments; Fleiss kappa on complete items = **0.383** over **52** fully judged claims.
- Agreement on `entailment`: mean pairwise observed agreement = **0.630** across items with at least 2 judgments; Fleiss kappa on complete items = **0.390** over **47** fully judged claims.
- Agreement on `score_correct`: mean pairwise observed agreement = **0.717** across items with at least 2 judgments; Fleiss kappa on complete items = **0.379** over **47** fully judged claims.

## Derived audit-value metric (claim usable + annotator score defensible)
- Strict pass = `claim_well_formed=yes` and `score_correct=yes`.
- Lenient pass = `claim_well_formed=yes` and `score_correct in {yes, partial}`.
- heuristic: strict pass **0.0%** (2 judged rows), lenient pass **0.0%**
- heuristic_ann_gpt41mini: strict pass **57.1%** (49 judged rows), lenient pass **59.2%**
- llm_gpt5mini: strict pass **0.0%** (2 judged rows), lenient pass **50.0%**
- llm_gpt5mini_ann_gpt41mini: strict pass **74.5%** (47 judged rows), lenient pass **78.7%**
- llm_gptoss: strict pass **0.0%** (1 judged rows), lenient pass **0.0%**
- llm_gptoss_ann_gpt41mini: strict pass **66.7%** (48 judged rows), lenient pass **70.8%**

## Interpretation
- The human annotations are most valuable if they help us answer: are low-score rows truly useful negatives? The answer is still leaning **no** when claims are malformed or when humans disagree about whether the low score reflects lack of entailment versus low-value phrasing.
- The blank downstream fields are not a reason to skip agreement analysis; they instead show that some annotators are using a two-stage logic: malformed claim => stop. That is itself a meaningful data-quality signal.
- For project decisions, the most useful outputs from round3 will likely be: (a) majority human decision per claim, (b) per-combo pass rates, (c) disagreement clusters on exact-match / vague / malformed claims.

## Recommended next analysis steps
1. Merge all five annotators into a canonical long-form table with a normalized missing-data policy.
2. Build claim-level majority labels for `claim_well_formed`, `entailment`, and derived pass/fail.
3. Analyze disagreement by combo and score bin, especially low-score rows.
4. Review the overlap 5 rows together with earlier A1/A2 labels to see whether round3 resolves the contested cases or confirms rubric ambiguity.