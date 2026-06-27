# ModernBERT UQ — classification / calibration experiments

This branch extends the base ModernBERT-UQ repo (branched from `main` `ec42f46`,
additions and edits only) with extra modeling heads, the RuleFollower / misinfo
experiment suite, CSCS run scripts, and a data-quality audit under `results/audit/`.

---

## 1. What's in this branch

318 files vs `main` (306 added, 12 modified). By area:

### `src/modernbert_uq/` — modeling
Core stages (`config`, `data`, `modeling`, `losses`, `evaluate`, `predict`, `train`,
`uncertainty`, `calibration`, `utils`) extended to support the heads below, plus two new
sub-packages:
- `distribution/` — distributional score head (soft targets / sigma variants).
- `ordinal/` — ordinal (CORN) score head.

**Supported approaches:** binary classification + difficulty regression, score regression,
distribution head, ordinal/CORN, and **Bradley-Terry pairwise ranking + ensemble**, with
MC-dropout and head-ensemble uncertainty.

### `configs/` — experiment suite (~70)
- `rulefollower_*` — the main suite across parser/annotator branches (`reg` / `dist` /
  `corn` / `ge8` / `gt8`), with seed, sigma, mc20, diff-on and smoke variants.
- `misinfo_*` — misinfo pilot.
- `bt_*`, `reg_base`, `reg_ensemble` — pairwise / ensemble variants.

### `scripts/` — data prep + CSCS runners
`prepare_*` (build / rebalance the training branches, BT pairs), `submit_*_cscs.sh` /
`train_*_cscs_container.sh` (SLURM drivers), `calibrate_*` / `evaluate_*` / `summarize_*`
(post-hoc calibration, MC-dropout eval, result summaries).

### `tests/`
Unit suite for every stage + the new heads. **41 unit tests pass**; 3 end-to-end smoke
tests need a ModernBERT download and are not run here.

### `results/audit/` — data-quality audit
Code, docs, and analysis results for the audit of the training data (see §3). Human
annotation **distribution / raw label files are excluded**; analysis outputs, the
labeling/auditor code, and the write-ups are kept.

---

## 2. Pipeline

```
config (YAML)  →  load + tokenize data  →  train (head per config)  →  predict / MC-dropout  →  evaluate  →  calibrate
```

A run is driven entirely by a `configs/*.yaml` via `scripts/*train*` (locally) or the
`submit_*_cscs.sh` SLURM drivers (cluster). Outputs land under `outputs/<branch>/`.

---

## 3. Data-quality audit (`results/audit/`)

Audits whether the RuleFollower (claim, source, score) training data is trustworthy enough
to train/calibrate the scorer. Stages: **A** programmatic checks, **A.5** claim filter,
**B** tiered human + LLM-auditor claim audit, **D** branch triage, plus synthetic-negative
generation. Headline findings (full write-up in `results/audit/audit_report.md`):

- **Parser:** heuristic parser is the noisiest (~47% lexically broken claims); the
  `llm gpt-5-mini × gpt-4.1-mini` combo evaluates cleanest. Retire heuristic for primary use.
- **Score bins:** score-8/9 labels are reliable (~91–100% truly entailed); **score-1/3
  labels are not** (0–30% acceptance) — the natural negative class can't be used as-is
  (hence the synthetic-negative path in `synth/`).
- **Splits:** no `text_id` leakage across train/val/test on any branch.
- All Stage-B numbers are LLM-auditor-scored against a human calibration set; see the
  per-stage `summary.md` files.

> Inter-annotator analysis files retain annotator names (internal team). The blank labeling
> packs and per-annotator raw returns are not included.

---

## 4. Data & compute

- **Inputs:** RuleFollower parsed claims (the `rulefollower_*` branches under `data/`,
  derived from the RuleFollower pipeline / HF `RuleFollower/rulefollower_parsed`).
- **Outputs:** training runs were produced on **CSCS Clariden**; bulk outputs (checkpoints,
  per-run prediction dumps) are **not in git**. BT/regression refinement metrics are
  summarized in the audit and reproducible from the configs.

---

## 5. Known notes

1. **BT loss test fixed:** `tests/test_bt_loss.py::test_bt_loss_gradients_flow` previously
   asserted the rejected-item gradient was negative; for standard Bradley-Terry it is
   positive (the loss is correct). Assertion + comment corrected. BT runs were verified on
   CSCS (positive score correlations across conditions).
2. **Large outputs excluded** (checkpoints / prediction dumps); reproduce from configs.
3. **Human annotation distribution files excluded** from `results/audit/`; analysis results
   and the labeling/auditor code are kept.
