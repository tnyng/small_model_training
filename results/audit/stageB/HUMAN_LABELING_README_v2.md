# Human labeling pack v2 (second annotator)

Second annotator: please hand-label the 30 rows in `human_labels_v2_blank.csv` using the rubric below. Do **not** open the other CSVs in this folder until you finish — they contain Claude's and the first annotator's judgments and would bias your labeling.

## How to label

For each row, fill these four columns:

### `claim_well_formed`: `yes` or `no`
Is the `claim` a single atomic proposition (subject + predicate, has a truth value, readable on its own)?
- `yes` — clean atomic claim, even if awkwardly phrased
- `no` — sentence fragment, multiple stitched-together claims, scratchpad reasoning ("i should look for…"), question, hypothetical, or unintelligible without the source

### `entailment`: `yes`, `partial`, or `no`
Does the `source` entail the `claim`?
- `yes` — the source supports the claim. Trivial-but-faithful paraphrase counts as yes (e.g. *"the text mentions Trump"* on a Trump article = `yes`).
- `partial` — grounded but distorted: a qualifier was dropped, claims merged, stance/intensity added, or quantifier shifted.
- `no` — claim content not in source, contradicts source, hallucinated.

If you can't find the supporting span in 30 seconds, mark `partial` or `no`.

### `score_correct`: `yes`, `no`, or `partial`
Given the entailment label you just gave, is `score_annotator` (the column with the original 1–9 score) defensible *given the project goal* (this data is for training an entailment verifier)?

- `entailment = yes` → expected score 8–9 → if `score_annotator` is in [8,9], `yes`; else `no` (or `partial` if close)
- `entailment = partial` → expected 4–7 → in [4,7], `yes`; else `no` (or `partial`)
- `entailment = no` → expected 1–3 → in [1,3], `yes`; else `no` (or `partial`)

**Important nuance**: if the claim is essentially a *verbatim copy of the source* with a "the text says X" prefix (i.e. not really a claim, just a quote), the annotator's *low* score is correct under the project goal — those rows are noise. Mark `score_correct = yes` (or `partial`) if the annotator gave it a low score.

### `vagueness`: `yes` or `no` (auxiliary, optional)
Is the claim too generic to be informative even if faithful? (e.g. *"the text discusses immigration"* — faithful but vague)

### `notes`: free text (optional)
Anything noteworthy: a borderline call, a contested judgment, a finding to flag.

## Important reminders

- **Judge each field independently.** Don't infer entailment from coherence or vice versa.
- **Do NOT use `score_annotator` as evidence** for entailment or claim_well_formed. That would create circularity.
- **The `source` column has the full original text** (median ~2 400 chars). Read it for entailment judgments.

## When you're done

1. Save your filled CSV as `human_labels_v2_filled.csv` in this folder.
2. Run:
   ```
   python3 modernbert_uq/results/audit/stageB/merge_v2_labels.py
   ```
3. The script outputs:
   - **You vs Claude** kappa (on all 30 rows).
   - **You vs first annotator** kappa (on the 5 overlap rows).
   - **All three** combined accuracy under user's interpretation, per combo.

The 5 overlap rows are intentionally drawn from cases where Claude and the first annotator disagreed — your independent judgment helps resolve those.
