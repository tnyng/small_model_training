# Human labeling pack (Stage B Tier 1 calibration)

You're going to hand-label 30 rows so we can compute agreement against Claude's labels and validate the Stage B numbers.

## Files

- `human_labels_blank.csv` — the 30 rows to label. Has the claim, source, and the original annotator's score. **Empty columns for you to fill in.** No Claude labels visible.
- `claude_labels_for_comparison.csv` — Claude's labels keyed by `row_id`. **Don't open this until you finish hand-labeling.** It's the answer key.
- `merge_human_labels.py` — after you finish, runs Cohen's kappa per field and per score bin, and dumps the disagreements.

## How to label

Open `human_labels_blank.csv`, fill these four columns for each row:

### `claim_well_formed`: `yes` or `no`

Is the claim a single atomic proposition (subject + predicate, has a truth value, readable on its own)?
- `yes` — clean atomic claim, even if awkward
- `no` — sentence fragment, multiple stitched-together claims, scratchpad reasoning ("i should look for…"), question, hypothetical, or unintelligible without the source

### `entailment`: `yes`, `partial`, or `no`

Does the source entail the claim?
- `yes` — the source supports the claim. Trivial-but-faithful paraphrase counts as yes (e.g. *"the text mentions Trump"* on a Trump article = `yes`).
- `partial` — grounded but distorted: a qualifier was dropped, two distinct source claims were merged, stance/intensity was added, or quantifier shifted.
- `no` — claim content not in source, contradicts source, hallucinated.

Tip: if you can't find the supporting span in 30 seconds, it's probably `partial` or `no`.

### `score_correct`: `yes` or `no`

Given the entailment label you just gave, is `score_annotator` (the column with the annotator's 1–9 score) in the *expected range*?

- `entailment = yes` → expected score 8–9 → if `score_annotator` is in [8,9], `score_correct = yes`; else `no`.
- `entailment = partial` → expected score 4–7 → if in [4,7], `yes`; else `no`.
- `entailment = no` → expected score 1–3 → if in [1,3], `yes`; else `no`.

### `vagueness`: `yes` or `no` (optional, auxiliary)

Is the claim too generic to be informative even if faithful? (e.g. *"the text discusses immigration"* — faithful but vague)

### `notes`: free text (optional)

Anything noteworthy: a borderline call you weren't sure about, a reason for a particular decision, a finding you want to flag.

## Important reminders

- **Judge each field independently.** Don't infer entailment from coherence or vice versa.
- **Do NOT use the assigned `score_annotator` as evidence** when judging entailment or claim_well_formed. That would create circularity. Score is judged separately under `score_correct`.
- **Look at the source span before scoring entailment.** If you can't find it quickly, mark `partial` or `no`.
- The `source` column is truncated to ~300 chars. If a claim's content might be in the truncated part and not visible, give `partial` rather than `no`.

## When you're done

1. Save your filled CSV as `human_labels_filled.csv` in the same folder.
2. Run:
   ```
   python3 modernbert_uq/results/audit/stageB/merge_human_labels.py
   ```
3. The script outputs:
   - Cohen's kappa per field (overall and per score bin) — agreement against Claude.
   - The full list of disagreements with both your notes and Claude's note, written to `human_vs_claude_disagreements.csv`.

## What the numbers mean

- kappa ≥ 0.8 — strong agreement; Claude's labels can be trusted as Tier 2.
- kappa 0.6–0.8 — moderate agreement; Claude's labels are usable but disagreements should be inspected.
- kappa < 0.6 — poor agreement; Claude's Stage B numbers need revision based on your labels.

The per-score-bin breakdown is the diagnostic for "is the low-score-region label-faithfulness problem real or did Claude over-call it?" — if we agree on score 8–9 but disagree on score 1–4, that points at exactly which judgments are robust and which aren't.
