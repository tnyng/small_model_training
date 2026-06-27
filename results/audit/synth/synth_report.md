# Synthetic Negatives — Generation Report

## Inputs

| | |
|---|---|
| Positives source | `llm gpt-5-mini × gpt-4.1-mini reg_base`, post-filter, score 8/9 |
| Splits used | train + validation |
| Test split | held out, untouched |

---

## Results

| | Value |
|---|---:|
| Positives processed | 11 383 |
| Negatives requested (2 per positive) | 22 766 |
| **Negatives accepted** | **22 154** (97.3%) |
| Failures | 612 (2.7%) |

---

## Edit-type distribution

| Type | Count | % |
|---|---:|---:|
| entity (swap) | 6 164 | 27.8 |
| fact (substitution) | 5 784 | 26.1 |
| stance (flip) | 4 949 | 22.3 |
| negation | 3 531 | 15.9 |
| number | 1 226 | 5.5 |
| quantifier | 500 | 2.3 |

Good diversity — no collapse to one easy edit type.

---

## By split

| Split | Positives | Synthetic negatives | Total labeled pairs |
|---|---:|---:|---:|
| train | 9 328 | 18 111 | 27 439 |
| validation | 2 055 | 4 043 | 6 098 |
| test (held out) | 1 916 | — | — |

---

## Sanity checks

| Check | Result |
|---|---|
| Edits identical to original claim | 0 |
| Edits with 0% source-token coverage | 0 / 500 sampled |
| Edit length vs original | typically 1–3 word changes |

---

## Sample edits

| Original | Edited | Type |
|---|---|---|
| the text quotes **Morris** describing Priebus reaching out to Latino immigrants | the text quotes **Ryan Mahoney** describing Priebus reaching out to Latino immigrants | entity |
| **Quetta** is described as the headquarters of the exiled Afghan Taliban leadership | **Karachi** is described as the headquarters of the exiled Afghan Taliban leadership | entity |
| numerous **Republicans** considering 2016 presidential campaigns | numerous **Democrats** considering 2016 presidential campaigns | entity |
| courts are being **asked** to do work that Congress might otherwise do | courts are **refusing** to do work that Congress might otherwise do | stance |
| Morris **declining** to provide details about the ad | Morris **agreeing** to provide details about the ad | stance |
| Morris's contract was **not renewed** | Morris's contract was **renewed** | negation |
| Trump uses courts to **solidify** his conservative coalition | Trump uses courts to **dismantle** his conservative coalition | stance |
