# Audit Summary: 3 LLM Auditors vs 5 Humans

## Setup

- Humans: `Yang`, `Yu`, `Alexander`, `Jingwei`, `Tobi`
- LLM auditors: `Opus 4`, `Sonnet 4.6`, `Codex / GPT-5.4`

## Human-Only Agreement

| Field | Pairwise Agreement | Fleiss' Kappa |
|---|---:|---:|
| `claim_well_formed` | 0.877 | 0.373 |
| `entailment` | 0.627 | 0.308 |
| `score_correct` | 0.706 | 0.359 |
| `pass_strict` | 0.714 | 0.379 |
| `pass_lenient` | 0.745 | 0.459 |

Interpretation:

- Humans agree strongly on `claim_well_formed`
- `entailment` is the hardest field

## Model vs 5 Humans: Pairwise Agreement

| Field | Codex / GPT-5.4 | Opus 4 | Sonnet 4.6 |
|---|---:|---:|---:|
| `claim_well_formed` | **0.908** | 0.850 | 0.837 |
| `entailment` | 0.664 | **0.684** | 0.493 |
| `score_correct` | 0.603 | 0.576 | **0.636** |
| `pass_strict` | 0.747 | **0.753** | 0.719 |
| `pass_lenient` | 0.644 | 0.630 | **0.678** |

Interpretation:

- `Codex / GPT-5.4` is best overall
- `Opus 4` is best on entailment
- `Sonnet 4.6` is weakest on entailment

## Model vs 5 Humans: Mean Cohen's Kappa

| Field | Codex / GPT-5.4 | Opus 4 | Sonnet 4.6 |
|---|---:|---:|---:|
| `claim_well_formed` | **0.539** | 0.528 | 0.406 |
| `entailment` | 0.389 | **0.438** | 0.268 |
| `score_correct` | **0.250** | 0.192 | 0.100 |
| `pass_strict` | 0.531 | **0.554** | 0.473 |
| `pass_lenient` | 0.344 | 0.347 | **0.370** |

Interpretation:

- same pattern as raw agreement
- Codex strongest overall
- Opus strongest on entailment

## Fleiss' Kappa: Humans Only vs Humans + Model

| Field | Humans Only | + Codex | + Opus 4 | + Sonnet 4.6 |
|---|---:|---:|---:|---:|
| `claim_well_formed` | 0.394 | 0.479 | **0.519** | 0.448 |
| `entailment` | 0.265 | 0.295 | **0.323** | 0.236 |
| `score_correct` | **0.358** | 0.286 | 0.241 | 0.261 |
| `pass_strict` | **0.733** | 0.620 | 0.646 | 0.603 |
| `pass_lenient` | **0.556** | 0.400 | 0.418 | 0.449 |

Interpretation:

- `Opus 4` behaves like the best extra panel member on the core semantic fields
- adding any model hurts agreement on `score_correct` and pass-style fields

## Main Takeaway

The human panel already suggested that:

- high-score rows are relatively trustworthy
- low-score rows are much noisier

The extra LLM auditors reinforce this rather than overturn it:

- strong agreement on higher-confidence, well-formed rows
- weaker agreement on low-score and semantically ambiguous rows

