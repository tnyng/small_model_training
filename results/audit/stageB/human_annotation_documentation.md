# Human Annotation — Project Documentation

This documents the human-annotation component of the data audit. It is intended as a methodology reference for the project / paper.

## 1. Purpose

To validate the labels produced by the LLM annotator (`gpt-4.1-mini`) on a stratified sample of `(source, claim, score)` triples. The audit's outputs are used to:

- estimate label-faithfulness accuracy per `(parser, annotator)` combo,
- compute inter-rater agreement (Cohen's kappa) for headline-metric defensibility,
- identify failure modes in the raw audit data.

The results feed directly into the parser-pick decision and the negative-class-handling strategy for the RFChecker training corpus.

## 2. Sample

| Round | Annotators | Rows / annotator | Score-bin stratification | Total unique rows | Coverage per row |
|---|:---:|:---:|---|:---:|:---:|
| 1 | 1 (project lead) | 30 | 10 / 5 / 10 / 5 (1–4 / 5–7 / 8–9 / flag) | 30 | 1× human + 1× LLM |
| 2 | 1 (second annotator) | 30 | 25 new + 5 overlap with round 1 | 25 new (5 overlap re-used) | up to 2× human + 1× LLM |
| 3 | 5 (project lead, second annotator, 3 new) | 30 each = 150 total slots | 10 / 10 / 10 (1–4 / 5–7 / 8–9), with ±1 deviation for new annotators | 53 (48 new + 5 overlap from earlier rounds) | 3× human per row (52 rows) + 4× human (1 row); existing rounds carry over for the 5 overlap rows |

All rows are drawn from three audited `(parser, annotator)` combos sharing the same annotator (`gpt-4.1-mini`), so any cross-parser difference reflects parser variation alone.

## 3. Annotation rubric

Each row is annotated on three independent fields:

| Field | Values | Question |
|---|---|---|
| `claim_well_formed` | `yes` / `no` | Is the claim a single complete proposition that has a truth value? |
| `entailment` | `yes` / `partial` / `no` | Does the source entail the claim? |
| `score_correct` | `yes` / `partial` / `no` | Given the entailment label, is the original annotator's 1–9 score in the expected range? |

A fourth field, `vagueness`, was collected in rounds 1–2 as an auxiliary diagnostic but is not part of the headline metric. It was dropped for round 3.

### `score_correct` decision rule

The original annotator was prompted to score *"how accurately does the description describe the text"* on a 1–9 scale. The expected mapping from human entailment label to original 1–9 score is:

| Entailment | Expected score range |
|---|:---:|
| `yes` | 8–9 |
| `partial` | 4–7 |
| `no` | 1–3 |

A row's `score_correct` is `yes` if the original score falls in the expected range, `partial` if one bin off, and `no` if clearly outside.

## 4. Headline metric — label-faithfulness accuracy

A row is **accepted** iff `claim_well_formed = yes` AND `score_correct = yes`. Per-combo label-faithfulness accuracy is the fraction of accepted rows in the audited sample for that combo.

Two summary views are reported in the audit:

- **strict** — only `score_correct = yes` counts toward acceptance.
- **lenient** — `score_correct = yes` or `partial` counts.

## 5. Independence and rubric-discipline rules

Annotators apply the rubric blindly:

- The original annotator's 1–9 score is visible in the row, but **must not** be used as evidence when judging `entailment` or `claim_well_formed`. It is only used to compute `score_correct`.
- Annotators do not see other annotators' labels for the same row.
- The auditor LLM (a different model from any project parser/annotator) labels the same rows independently, providing a third source of agreement data.

## 6. Inter-rater agreement protocol

- Round 2 introduced **5 overlap rows** with round 1 to enable inter-human kappa on contested cases.
- Round 3 uses a balanced incomplete design: each new claim is labeled by exactly 3 of the 5 annotators; the 5 round-1/2 overlap rows receive a third human label from one of the new annotators.
- Cohen's kappa is computed per field, both overall and per score-bin, so disagreement is decomposed by where it occurs (e.g., low-score bin disagreement isolates the contested-rubric pattern described in the audit team report).

## 7. Sample size and CI caveats

At 30 rows per combo (rounds 1–2), the 95 % CI on a label-faithfulness estimate near 60 % is approximately ±18 pp; at the round-3 size of ~50 rows per combo it falls to approximately ±13 pp. The cross-parser-family gap (heuristic vs LLM) exceeds the CI in every view; the within-LLM-family gap (≤7 pp across views) sits inside the CI and is reported as directional only.

## 8. Pre-registered design points

To prevent post-hoc threshold tuning, the audit's decision rules — keep / caveat / drop thresholds, parser-decision triggers, falsification thresholds for the project's working hypotheses — were locked in writing before any human annotation began. The locked rule set is in `audit_decision_rules.md`.
