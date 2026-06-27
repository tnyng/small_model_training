# Annotation Guideline

## What's in a row

| Column | Meaning |
|---|---|
| `row_id` | Identifier — leave alone |
| `combo` | Which parser/annotator combo produced this row — leave alone |
| `score_annotator` | The original annotator's 1–9 score — **read this**, you'll evaluate it in `score_correct` |
| `claim` | The claim being judged |
| `source` | The source text the claim is supposed to describe |

---

## Column 1 — `claim_well_formed`: `yes` or `no`

Question: **Is the claim a single, complete proposition that has a truth value?**

- `yes` — one subject + one predicate, makes sense on its own, you could tell whether it's true or false (with the source in hand). Even awkward grammar is fine if it's a real proposition.
- `no` — sentence fragment (no main predicate), multiple stitched-together claims, scratchpad reasoning ("i should look for…"), a question, a hypothetical ("what if X?"), or unintelligible without the source.

Examples:

| Claim | claim_well_formed |
|---|---|
| *"the text mentions Trump's stance on immigration."* | yes |
| *"the text the article mentions Rep."* | no (fragment) |
| *"i should look for any words indicating politeness."* | no (scratchpad) |
| *"are these claims supported by facts?"* | no (question) |
| *"the text is discussing political bias."* (awkward but a proposition) | yes |

---

## Column 2 — `entailment`: `yes`, `partial`, or `no`

Question: **Does the source entail the claim? (i.e., if I read the source, can I conclude the claim is true?)**

- `yes` — the source clearly supports the claim. You can locate the supporting span in the source within ~30 seconds. Faithful paraphrase counts.
- `partial` — grounded but distorted. E.g., a qualifier got dropped ("often" → "always"), two source claims got merged, stance/intensity was added or amplified, or a quantifier shifted ("most" → "all").
- `no` — claim content not in the source, or contradicts the source, or hallucinated.

Examples:

| Source | Claim | entailment |
|---|---|---|
| "There is no Planet B." | *"the text uses the phrase 'There is no Planet B'."* | yes |
| "Most studies show X." | *"the text states all studies show X."* | partial |
| "Trump tweeted about taxes." | *"the text describes a Jungkook concert."* | no |
| "Some experts disagree." | *"the text states experts disagree."* (qualifier dropped) | partial |

**Important:** judge `entailment` *without* using `score_annotator` as evidence. They're judged independently.

---

## Column 3 — `score_correct`: `yes`, `partial`, or `no`

Question: **Given your `entailment` answer, is `score_annotator` in the expected range?**

The original annotator was supposed to give scores roughly:
- entailment `yes` → expected score **7–9**
- entailment `partial` → expected score **3–7**
- entailment `no` → expected score **1–3**

So:
- `yes` — `score_annotator` is in the expected range for your entailment label.
- `no` — clearly outside the expected range (e.g., entailment yes but score 2, or entailment no but score 9).
- `partial` — borderline / one bin off (e.g., entailment yes but score 7, or entailment partial but score 3 or 8).

Examples:

| your `entailment` | `score_annotator` | `score_correct` |
|---|:---:|---|
| yes | 9 | yes |
| partial/yes | 7 | partial/no |
| yes | 2 | no |
| partial | 5 | yes |
| partial | 1 | no |
| partial/no | 3 | partial/no |
| no | 2 | yes |
| no | 8 | no |

---

## A few practical reminders

- **Judge each column independently.** Don't infer entailment from claim_well_formed, or score_correct from entailment without checking the score.
- **Don't peek at `score_annotator` when judging entailment.** That would create circularity. Your entailment judgment should be based only on (claim, source).
- **Source text can be long.** If a claim's content might be in a part of the source you didn't read carefully, give `partial` rather than `no`.
- **Notes column is optional.** Use it for borderline calls or anything you want to flag for discussion later.
