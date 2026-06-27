# Parser Decision (Stage C)

## Decision

**Outcome B — main + secondary**:

- **Main parser**: `llm gpt-5-mini`
- **Secondary** (robustness / appendix only): `llm gpt-oss`
- **Dropped as primary training source**: `heuristic` (kept for robustness comparisons only)

## Evidence

### Stage B label-faithfulness accuracy (Stage 0 rule input)

Four views of the metric, all evaluated under the user's interpretation of `score_correct` (verbatim-replicate-as-claim is noise that low scores correctly flag):

| View | heuristic | llm gpt-5-mini | llm gpt-oss |
|---|---:|---:|---:|
| User-only strict (10/combo) | 60% | 56% | 70% |
| User-only lenient (10/combo) | 70% | 78% | 80% |
| Combined strict (30/combo) | 47% | 67% | 57% |
| Combined lenient (30/combo) | 50% | 73% | 60% |

Pairwise gap pattern:

- llm gpt-oss vs heuristic: **+10 pp in every view** (consistent)
- llm gpt-5-mini vs heuristic: **+8, +20, +23 pp in three of four views**; −4 pp in the noisiest view (user-only strict, ±28 pp CI)
- llm gpt-5-mini vs llm gpt-oss: not statistically separable across views; gpt-oss slightly ahead in low-N user-only views, gpt-5-mini ahead in larger-N combined views.

### Stage A.5 drop rate (independent signal)

Fraction of raw rows the lexical filter excluded (i.e. claims that are not atomic well-formed propositions):

| Combo | drop rate |
|---|---:|
| heuristic | **47%** |
| llm gpt-5-mini | 25% |
| llm gpt-oss | 34% |

Heuristic produces nearly 2× more lexically broken claims than llm gpt-5-mini before any labeling question.

### Cross-parser source coverage (Stage A5)

The 5 main combos sit on essentially the same set of ~1 470 source articles (Jaccard ≥ 0.99). Parser-quality comparisons are well-grounded — same articles, different parser outputs.

### Heuristic produces verbatim-replicate claims

The user's hand-labeling session surfaced that heuristic emits claims like *"the text says: 'cancer patients live 4x longer by refusing chemotherapy'"* — verbatim quotation of the source rather than an extracted claim. The annotator correctly assigns these low scores, so labels are faithful in that sense, but the *claim* is still not useful as RFChecker training data. This counts against heuristic regardless of how `score_correct` is interpreted.

## Why Outcome B and not Outcome C (keep all three)

The proposal calls for "different reasoning models" producing claims (Appendix A.2) — so parser diversity is a stated project value. Outcome C would satisfy that with all three parsers. We don't pick C because:

- Heuristic loses in **both** independent quality signals: Stage A.5 drop rate (47% vs 25–34%) and Stage B label-faithfulness (consistent in 7 of 8 pairwise comparisons).
- Parser-diversity is already satisfied by keeping two LLM parsers (gpt-5-mini, gpt-oss).
- Heuristic remains available for robustness comparisons / ablations as a *secondary* (Outcome B), not dropped entirely.

## Why gpt-5-mini as main rather than gpt-oss

Honest answer: the Stage B accuracy numbers don't separate them at this sample size (CI ~ ±28 pp on 10 rows; ~ ±18 pp on 30). The pick is based on:

- gpt-5-mini has more rows of training data (`reg_base` train ~ 13 600) than the gpt-oss probe combos (1 050–1 500 train) — more usable training pairs at scale.
- gpt-5-mini sits on the full ~1 470-article corpus; gpt-oss-annotator combos sit on a much smaller (~111-article) subset.
- llm gpt-oss + gpt-4.1-mini *does* sit on the full corpus and could be a strong main candidate, but gpt-5-mini is preferred for its larger Stage B sample evidence at full scale.

If a future audit shows gpt-oss is materially better, swap them.

## Caveats

1. **Sample size is small.** 10 rows per combo (user-only view) or 30 rows per combo (combined view). 95% CI on 60% from 10 rows is ~ ±28 pp. The picks within "the LLM family" are not statistically defensible — the heuristic-vs-LLM-family decision is.
2. **The combined-view numbers blend the user's 30 verified labels with Claude's 60 unverified labels** under a slightly different interpretation. The user-only view is more reliable but noisier.
3. **Decision is provisional** and should be revisited if (a) you hand-label another 60 rows, or (b) Stage F calibration reveals parser-specific reliability differences.

## Implications

- Stage D (branch triage) drops heuristic-parser branches as primary, keeps them as robustness comparisons.
- For the paper: main results are reported on llm gpt-5-mini branches. llm gpt-oss in an appendix as parser-robustness check. heuristic mentioned briefly as an early-pipeline pre-curated baseline that under-performed on data quality.
- For future RFChecker training: the cleaned llm gpt-5-mini branches are the candidate training corpus.
- Hard-negative examples (1–4 score) remain unreliable across all parsers and should be augmented synthetically per the proposal Appendix A.2.

## What this does NOT decide

- The Stage F calibration question (does temperature scaling work on llm gpt-5-mini branches): pending.
- Whether to use BT (Bradley-Terry) pairs or regression labels: defers to Stage E + existing modelling results.
- The annotator decision (gpt-4.1-mini vs gpt-5-mini): not addressed in this audit; the cross-annotator comparison on the same heuristic claims remains a useful follow-up.
