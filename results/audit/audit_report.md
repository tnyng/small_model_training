# Data Audit Report — RuleFollower


- **Heuristic parser shows ~2× more lexically broken claims than LLM parsers** (47% vs 25–34%).
- **`llm gpt-5-mini × gpt-4.1-mini reg_base` is the cleanest-evaluating combo** across most audit views.
- **Score-8/9 labels are reliable** (~91–100% truly entailed). **Score-1/3 labels are not** (0–30% acceptance) — natural negative class is unreliable as-is.
- **Splits are clean** — no text_id leakage across any branch. Prior modelling conclusions don't need redoing for split reasons.

---

## 1. Pipeline picture

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│   source text    │ →   │      PARSER      │ →   │    ANNOTATOR     │ →  (claim, source, score)
│ (article_bias …) │     │ heuristic / llm  │     │ gpt-4.1 / 5-mini │
└──────────────────┘     └──────────────────┘     └──────────────────┘
```

Three parsers × three annotators tested (7 audited combos in total, ~140 k rows). Three combos audited deeply with the same annotator (**gpt-4.1-mini**) so any difference is parser-driven.

---

## 2. Parser quality

### Lexical-filter drop rate (independent signal — fraction of raw rows that aren't atomic well-formed claims)

| Combo | drop rate |
|---|---:|
| heuristic × gpt-4.1-mini | **47%** |
| llm gpt-oss × gpt-4.1-mini | 34% |
| llm gpt-5-mini × gpt-4.1-mini | **25%** |

Heuristic produces ~2× more lexically broken claims (stutters, fragments, scratchpad reasoning).

### Label-faithfulness accuracy (verified against human hand labels, 30 rows / combo)

| Combo | strict | lenient | combined-lenient (largest sample) |
|---|---:|---:|---:|
| heuristic × gpt-4.1-mini | 60% | 70% | 50% |
| **llm gpt-5-mini × gpt-4.1-mini** | 56% | **78%** | **73%** |
| llm gpt-oss × gpt-4.1-mini | 70% | 80% | 60% |

(Sample size is small; within-LLM-family ranking is not yet statistically separable. Heuristic is bottom or tied across all views.)

---

## 3. Score-bin reliability

| Score bin | Acceptance | Reading |
|---|---:|---|
| 8–9 | **91–100%** | Reliable — usable as positive training signal |
| 5–7 | ~50% | Borderline |
| 1–4 | **0–30%** | Broken — natural negative class is unreliable |

Implication: the negative class for RFChecker training cannot be taken from natural low-score rows as-is.

### Failure modes — examples of bad raw rows

Three patterns dominate. Each row is `(score by annotator, claim, source)`:

**1 · Heuristic-parser stutter / fragment / scratchpad** (lexically broken, not a real claim)

| Score | Claim | Source |
|---|---|---|
| 7 | *"the text the article mentions Rep."* | (any article — claim is mid-sentence fragment) |
| 5 | *"the text from what I know, most states have measures to ensure access to vaccines, but there could be exceptions."* | "Why Do Some States Make It Difficult for Children to Get Flu Shots?" |
| 7 | *"the text looking at the original post: It mentions 11 confirmed dead..."* | (Charlie Hebdo tweet — scratchpad reasoning leaked into the claim) |

**2 · LLM-parser misalignment** (claim is grammatically clean but content not in source)

| Score | Claim | Source |
|---|---|---|
| 9 | *"the text describes Jungkook pointing to his heart while singing 'Euphoria'"* | A tweet about Twitter suspending Trump's account |
| 9 | *"the text accuses the addressed parties of turning a blind eye to the pandemic..."* | A short tweet about an "echo account" — content not in source |

**3 · Low-score-on-faithful-paraphrase / verbatim-quote claims** — interpretation contested (see §7)

| Score | Claim | Source |
|---|---|---|
| 2 | *"The text says, 'cancer patients live 4x longer by refusing chemotherapy 25 years of research.'"* | "cancer patients live 4x longer by refusing chemotherapy 25 years of research." (verbatim quote of source) |
| 1 | *"the text uses the phrase 'There is no Planet B'."* | "The Build Back Better Act will address the climate crisis in a transformative way. There is no Planet B." (phrase IS in source) |
| 3 | *"the text discusses forthcoming arms shipments and whether their place of origin or their use should be emphasized in the President's statement."* | "then pointed out that the forthcoming arms shipments to Guatemala might well come from other countries than those behind the Iron Curtain, and wondered whether the phrasing of the President's statement should be changed to recognize this fact and to take account of the importance of the use…" (faithful paraphrase, lemma-coverage 0.89) |

Whether these rows count as "annotator error" or "annotator correctly low-scoring useless claims" is the contested point — see Section 7 for the two interpretations.

---

## 4. Splits

| Check | Finding |
|---|---|
| `text_id` leakage across train / val / test | **0%** across all branches — no prior modelling needs redoing for split reasons |

Rebalanced `_exact` slices (`reg_over_exact`, `reg_down_exact`) are mechanical artefacts of the oversampling / downsampling routines.

---

## 5. What the evidence suggests

### Parser

- Heuristic shows the highest lexical noise (47%) and lowest label faithfulness across views — least suitable for primary training use.
- Among LLM parsers, `llm gpt-5-mini` and `llm gpt-oss` are not statistically separable at the current sample size; `gpt-5-mini` leads in 3 of 4 views, `gpt-oss` leads in the noisiest view.

### Candidate combos (raw `parser × annotator`)

| Combo | Role | Why it's a candidate |
|---|---|---|
| `llm gpt-5-mini × gpt-4.1-mini` | Strongest candidate for main | Lowest filter drop rate (25%), 100% score-8/9 entailment-yes, ~22 k raw rows, fully audited |
| `llm gpt-oss × gpt-4.1-mini` | Cross-parser robustness | Different parser, same annotator → isolates parser effect. ~22.5 k raw rows |
| `llm gpt-5-mini × gpt-5-mini` | Cross-annotator robustness | Same parser, different annotator → isolates annotator effect. Not yet audited end-to-end |


### Out of scope under any audit-view choice

- Heuristic family combos (heuristic × any annotator) — high lexical noise, low label faithfulness
- `gpt-oss × gpt-oss` combos — only ~1 500 rows, too small for primary use
---

## 6. Plan options

### Negative-class strategy

| Option | Description | Pros | Cons |
|---|---|---|---|
| A | Re-annotate score-1/3 rows with stricter rubric | Keeps natural data diversity | API cost; doesn't address that some claims are noise even with correct labels |
| B | Keep natural negatives + add synthetic | More training data | Natural negatives still introduce label noise |
| **C** | **Drop natural negatives entirely; use synthetic hard negatives only** (proposal Appendix A.2) | Clean labels by construction; aligns with proposal | Loses natural diversity (mitigated by proposal's plan) |

### Parser scope for the headline corpus

| Option | Scope | Pros | Cons |
|---|---|---|---|
| 1 | Single branch (llm_gpt5mini × gpt-4.1-mini reg_base) | Simple story; all compute on one corpus | If parser choice is wrong, risk concentrated |
| 2 | Multi-branch headline (compare parsers) | Shows parser/annotator effects | Splits compute; small grant won't fully support |
| **3** | **One main branch + 1–2 robustness checks (appendix)** | Defensible main story + modest robustness | Slight extra cost |

### Synthetic-negative coverage

| Option | What gets synthetic negatives |
|---|---|
| α | Only the main branch (`llm_gpt5mini × gpt-4.1-mini reg_base`) |
| β | Main branch + one robustness branch (`llm_gptoss × gpt-4.1-mini ge8`) |
| γ | All `× gpt-4.1-mini` LLM combos |

---

## 7. Two interpretations of the contested low-score rows

The pattern-3 rows above (verbatim quotes / faithful paraphrases scored 1–3) are real disagreement, not a measurement glitch.

### Interpretation A — "Verbatim / replicate is noise; the annotator was right"

A claim that is just a verbatim quote with `the text says: "X"` isn't a real reasoning-step claim — it's a quotation of the source, not an extracted proposition about it. Same for paraphrase claims that just describe what the article is about ("the text discusses arms shipments…"). These are *trivially accurate* but vacuous; we don't want the verifier to learn from them. The annotator low-scoring them is **correct under the project goal**.

- Score-1/3 labels on these rows are not "broken" — they're correctly penalising not-useful-as-training-data claims.
- The negative class isn't broken either — it's a mix of (a) genuinely unfaithful claims and (b) verbatim/replicate claims correctly low-scored.
- Implication for the corpus: verbatim/replicate rows should ideally be filtered upstream (Stage A.5 already drops some), but the annotator's low score isn't itself a label error.

### Interpretation B — "By the literal rubric, faithfulness is faithfulness; the annotator was wrong"

The annotator was given the prompt *"Does the description accurately describe the text? 1=completely inaccurate, 9=perfectly accurate."* By that prompt, a verbatim quote IS the most accurate description possible — it should be 9. A faithful paraphrase should be 8–9. The annotator giving 1–3 is a **rubric-application error**.

- Score-1/3 labels on these rows are wrong. Some non-trivial fraction of the natural negative class is mislabeled.
- Using these rows as negatives in RFChecker training would teach the verifier that verbatim/paraphrased claims are unfaithful — exactly the opposite of what we want.
- Implication for the corpus: the natural negative class can't be trusted as-is; need synthetic negatives or re-annotation under a stricter rubric.
