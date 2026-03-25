# ModernBERT UQ

Small research repo for encoder-based binary classification with:

- a shared encoder backbone
- a binary classification head
- an optional difficulty regression head
- uncertainty estimation utilities
- post-hoc calibration utilities

The code is intentionally narrow and currently optimized for iterative research rather than a polished training CLI.

## Current Status

Implemented today:

- dataset loading from CSV / JSONL
- pooled encoder classifier with difficulty regression
- masked joint BCE + MSE loss
- native `accelerate` training loop utilities
- MC-dropout uncertainty inference
- classifier head ensemble inference/training path
- score-only and difficulty-aware calibration
- local tests that skip torch-dependent files automatically when `torch` is unavailable

Planned but not implemented yet:

- LoRA ensemble
- Laplace / Bayesian linear head approximation
- full config-driven CLI training entrypoint

Important: this repository has not yet been validated in a real `torch + transformers` environment end-to-end with an actual Hugging Face encoder model. Treat the torch-side code as implementation-ready but still requiring first-run integration checks.

## Repository Layout

Core modules:

- `src/modernbert_uq/config.py`: YAML config loading
- `src/modernbert_uq/data.py`: dataset loading, tokenization, collation
- `src/modernbert_uq/modeling.py`: base model and head-ensemble path
- `src/modernbert_uq/losses.py`: masked joint loss
- `src/modernbert_uq/train.py`: training and batch prediction helpers
- `src/modernbert_uq/uncertainty.py`: MC-dropout summaries
- `src/modernbert_uq/evaluate.py`: classification and regression metrics
- `src/modernbert_uq/calibration.py`: temperature scaling and logistic calibration
- `src/modernbert_uq/predict.py`: prediction export utilities

Tests:

- pure-Python tests: config, data loading, evaluation, calibration
- torch tests: modeling, training smoke, uncertainty smoke, end-to-end smoke

## Data Contract

Supported input fields:

- `text`
- `label`
- `difficulty`
- optional `id`
- optional `split`

Supported file types:

- `.csv`
- `.jsonl`

Example CSV:

```csv
id,text,label,difficulty,split
ex-1,"The sky is blue.",1,0.10,train
ex-2,"Penguins can fly.",0,0.80,train
ex-3,"Water boils at 100C at sea level.",1,0.20,validation
```

## Installation

Python 3.10+ is recommended.

Install the package in editable mode:

```bash
pip install -e .[dev]
```

For GPU usage, install a matching PyTorch build first from the official PyTorch instructions, then install this repo.

## What The Main APIs Do

Load config:

```python
from modernbert_uq.config import load_config

config = load_config("configs/default.yaml")
```

Build a model:

```python
from modernbert_uq.modeling import ModernBertUQModel

model = ModernBertUQModel(
    model_name="answerdotai/ModernBERT-base",
    dropout_prob=0.1,
    uq_method="none",  # or "head_ensemble"
    num_ensemble_members=1,
)
```

Train using prepared batches:

```python
from modernbert_uq.train import TrainConfig, run_training

result = run_training(
    model=model,
    train_batches=train_batches,
    eval_batches=eval_batches,
    config=TrainConfig(
        learning_rate=2e-5,
        num_epochs=3,
        output_dir="outputs/run-001",
    ),
)
```

Run MC-dropout:

```python
from modernbert_uq.uncertainty import mc_dropout_predict

summary = mc_dropout_predict(model, batch, num_passes=8)
```

Fit a calibrator:

```python
from modernbert_uq.calibration import TemperatureScaler, FeatureLogisticCalibrator

temp = TemperatureScaler().fit(logits, labels)
difficulty_cal = FeatureLogisticCalibrator(use_difficulty=True).fit(
    logits=logits,
    labels=labels,
    predicted_difficulty=predicted_difficulty,
)
```

## Local Test Behavior

This repo is designed to stay partially testable on machines without `torch`.

Tests that require `torch` use the `requires_torch` marker and skip automatically when `torch` is not installed.

Run the non-torch suite:

```bash
pytest tests/test_config.py tests/test_data.py tests/test_evaluate.py tests/test_calibration.py -v
```

Run the full suite in a torch environment:

```bash
pytest tests/ -v
```

## Mini Test Matrix

This section is the recommended first validation pass once you have a real `torch + transformers` environment. Use a synthetic mini dataset and run each implemented algorithm in isolation before trying a larger experiment.

### 1. Prepare a synthetic mini dataset

Create `data/mini.csv`:

```csv
id,text,label,difficulty,split
ex-1,"Cats are mammals.",1,0.10,train
ex-2,"The moon is made of cheese.",0,0.95,train
ex-3,"Paris is in France.",1,0.10,train
ex-4,"Fish live on trees.",0,0.90,train
ex-5,"Water is wet.",1,0.20,validation
ex-6,"Fire is cold.",0,0.85,validation
```

Use a tiny encoder for first-run validation if you want the fastest smoke test. A practical choice is:

- `hf-internal-testing/tiny-random-bert`

If you want to test the intended backbone path directly, switch to:

- `answerdotai/ModernBERT-base`

### 2. Mini Test: Baseline dual-head classifier

Goal:

- confirm model instantiation works with a real HF encoder
- confirm one short training run finishes
- confirm prediction export and calibration work

Checklist:

- set `uq_method="none"`
- train for 1 epoch
- verify `training_state.pt` is written
- verify predictions include:
  - `logits`
  - `probabilities`
  - `difficulty_predictions`
- verify both temperature scaling and difficulty-aware calibration can fit on the exported predictions

Suggested command:

```bash
pytest tests/test_modeling.py tests/test_train_smoke.py tests/test_end_to_end_smoke.py -v
```

### 3. Mini Test: Head ensemble

Goal:

- confirm ensemble forward pass works on a real encoder
- confirm aggregated uncertainty fields are produced
- confirm the ensemble can train for a minimal smoke run

Configuration changes:

- `uq_method="head_ensemble"`
- `num_ensemble_members=3`
- keep `dropout_prob=0.0` for determinism during the first ensemble smoke test

Checklist:

- verify forward output includes:
  - `member_logits`
  - `mean_probability`
  - `probability_variance`
  - `predictive_entropy`
  - `mutual_information`
- verify `predict_batches` exports:
  - `uncertainty_method == "head_ensemble"`
  - `member_logits`
  - `probability_variance`
  - `mutual_information`

Suggested command:

```bash
pytest tests/test_modeling.py tests/test_train_smoke.py -v
```

### 4. Mini Test: MC-dropout

Goal:

- confirm dropout can be re-enabled during inference
- confirm repeated stochastic passes produce non-identical outputs
- confirm aggregated uncertainty summaries are numerically sane

Recommended setup:

- use the baseline model first
- set `dropout_prob=0.2` or higher
- run `mc_dropout_predict(..., num_passes=8)`

Checklist:

- verify `probability_samples` changes across passes
- verify these fields exist:
  - `mean_probability`
  - `probability_variance`
  - `predictive_entropy`
  - `mutual_information`
  - `mean_difficulty`
  - `difficulty_variance`

Suggested command:

```bash
pytest tests/test_uncertainty.py -v
```

### 5. Mini Test: Calibration

Goal:

- confirm the repo can fit post-hoc calibrators on exported model outputs

Checklist:

- fit `TemperatureScaler` on logits and labels
- fit `FeatureLogisticCalibrator(use_difficulty=False)`
- fit `FeatureLogisticCalibrator(use_difficulty=True)`
- verify all calibrated probabilities are in `[0, 1]`

Suggested command:

```bash
pytest tests/test_calibration.py tests/test_evaluate.py -v
```

### 6. Methods not ready for mini testing yet

These are part of the design direction but not implemented in the repo yet:

- `lora_ensemble`
- `laplace_linear_head`

Do not document them as supported runtime options yet, and do not treat the README as claiming they work today.

## Suggested First Real Validation Order

Use this order when you first run the repo in an environment with `torch` and `transformers` installed:

1. `pytest tests/test_modeling.py -v`
2. `pytest tests/test_train_smoke.py -v`
3. `pytest tests/test_uncertainty.py -v`
4. `pytest tests/test_end_to_end_smoke.py -v`
5. one manual mini-dataset run with a real Hugging Face encoder

If that passes, then move on to larger datasets or ModernBERT-specific runs.

## Notes

- `scripts/train.sh`, `scripts/evaluate.sh`, and `scripts/predict_mc_dropout.sh` are placeholders, not full production entrypoints.
- The current training utilities operate on already-prepared batches rather than a full config-driven experiment runner.
- The README is intentionally conservative: it only claims support for code paths that currently exist in the repository.
