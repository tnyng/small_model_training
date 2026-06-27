# Stage A.5 — Filter Rules (locked) and Validation

## Filter rules

All applied to the `property` column of each row. Each row gets a `drop` decision (excluded from filtered branches) or a `review` flag (kept but marked, for downstream Stage B inspection).

### Auto-drop

| Rule | Pattern / condition |
|---|---|
| `template_uses_phrase_no_cov` | regex `^the text uses the phrase` AND coverage = 0 (drop only when phrase isn't in source — a faithful "uses the phrase X" claim is a valid atomic positive for the RFChecker) |
| `template_vacuous_tag` | regex `^the text is (a tweet|relevant|irrelevant)\.?\s*$` |
| `template_stutter_the_text` | regex `^the text the\b` |
| `template_negative_no_cov` | regex `^the text does not (mention|reference|discuss|contain|include)` AND coverage = 0 |
| `awkward_the_text_is_x` | regex `^the text is (possible|likely|unlikely|maybe)` |
| `awkward_the_text_wh` | regex `^the text (what|how|when|where|why|whose|which)\b` |
| `awkward_the_text_that` | regex `^the text (that|this) (would|could) (imply|mean|suggest)` |
| `reasoning_leak_start` | starts with `i should`, `let me`, `looking at`, `without (knowing|more)`, `i need to`, `i would`, `i'm not sure` |
| `hypothetical_what_if` | contains ` what if ` |
| `ends_with_question` | property ends with `?` (after stripping trailing periods) |
| `rhetorical_right_q` | ends with `right?` |
| `too_short_lt5_tokens` | token count < 5 |
| `too_long_gt50_tokens` | token count > 50 |
| `coverage_zero` | lemma-coverage of claim tokens in source = 0 (and ≥ 4 content tokens) |

### Flag for review (do NOT auto-drop)

| Rule | Pattern / condition |
|---|---|
| `coverage_low` | lemma-coverage in (0, 0.10] |

`coverage` is a stop-word and meta-token-stripped substring match between claim content tokens and source — see `filter_claims.py`.

## Validation

Validated against `experiments/analysis/noisy_heuristic_claims_three_models_sample_20.csv` (20 user-labeled noisy claims, all with `noise_score` ≥ 2).

| Metric | Value | Target | Pass |
|---|---:|---:|---|
| Recall (drop only) | **80.0%** (16/20) | ≥ 70% | ✓ |
| Recall (drop + review) | **85.0%** (17/20) | — | — |

**Residual noise modes (the 3 missed)**: all involve hedge markers (`might`, `maybe`) embedded mid-sentence after `the text is X` or `the term X` constructions:
- N08: "The text is about interviews being fake, which **might** be a criticism..."
- N09: "the text is a statement against the ethno-state, so **maybe** it's not hate speech."
- N13: "The term 'cancer industry' **might** imply that the company is exploiting patients..."

These can't be safely caught by lexical rules without high false-positive risk on legitimate hedged claims. Documented as known limitation.

### Precision (spot-checked, not formally measured)

The labeled set has only positive examples (noisy), so precision can't be formally computed against it. Spot-checked manually against a stratified random sample of 150 claims from `heuristic + gpt-4.1-mini`:
- 86 drops, 4 review flags, 60 keeps.
- ~3 borderline drops in initial pass that I called false positives turned out to match the user's `awkward_start` + hedge pattern when re-read against the noise rubric.
- Net precision estimate: ≥ 0.90 — exceeds the Stage 0 target (0.85).

The full 150-claim sample with rule-fire annotations is in `random_drop_review.csv` if you want to spot-check more.

## Iterations

Stage 0 allows up to 3 iterations. Used 2:
1. Initial filter — recall 80%, but I was concerned about ~3 false positives in the random sample.
2. Tightened stutter and reasoning-leak rules with coverage thresholds — recall dropped to 55% (below target). The "false positives" turned out to match the user's labeled noise patterns; my caution was unwarranted. Reverted.

Final filter is the iteration-1 ruleset.
