# ModernBERT UQ Design

## Goal

Create a focused training repository for ModernBERT-like classifiers that supports native fine-tuning, single-node multi-GPU training, Monte Carlo dropout uncertainty quantification, and joint modeling of a binary target label with a difficulty regression label for downstream calibration experiments.

## Scope

The first version is a greenfield Python project in `small_model_training/` built on top of:

- PyTorch
- Hugging Face Transformers
- Accelerate
- Datasets / pandas for data ingestion
- scikit-learn for metrics and lightweight post-hoc calibration

The project is intentionally narrow. It should solve one problem well: fine-tune encoder classifiers with a shared backbone and two supervised heads, then evaluate whether predicted sample difficulty improves calibration over logits alone.

## Architecture

### Backbone and model

The core model uses a pretrained ModernBERT encoder loaded through `transformers`. A shared pooled representation feeds two task-specific heads:

- `classification_head`: binary classification logits for the target label
- `difficulty_head`: scalar regression output in `[0, 1]` for sample difficulty

The shared encoder is fully fine-tuned. The first version should avoid adapters or PEFT abstractions. The emphasis is native end-to-end fine-tuning.

### Forward contract

The forward pass should return a structured output containing:

- `logits_cls`
- `difficulty_pred`
- `loss`
- optional intermediate tensors needed for debugging or extension

The model must support batches where one of the labels is missing by applying loss masks. This keeps the data contract flexible while preserving a simple primary path where both labels are present.

### Joint loss

Training minimizes a configurable weighted sum:

- binary classification: `BCEWithLogitsLoss`
- difficulty regression: `MSELoss`

Formula:

`total_loss = cls_weight * cls_loss + difficulty_weight * difficulty_loss`

This is sufficient for the initial repo. More complex uncertainty-aware or heteroscedastic losses can be deferred until there is evidence they improve the task.

## Multi-GPU training

The primary training path should use a native PyTorch loop orchestrated by `accelerate`. This is the current best-fit approach for single-node multi-GPU support in the Transformers ecosystem because:

- `Trainer` is powered by Accelerate, but a custom loop is easier for multi-head outputs and uncertainty evaluation
- `accelerate launch` is the standard launcher for local multi-GPU training
- mixed precision, gradient accumulation, distributed dataloading, checkpointing, and device placement are all handled cleanly

The first version targets:

- single-node multi-GPU
- DDP-style distributed training via Accelerate
- optional mixed precision
- gradient accumulation
- deterministic seeding where practical

FSDP or DeepSpeed should not be part of the initial implementation unless memory constraints force them later.

## Uncertainty quantification

Monte Carlo dropout is the uncertainty method for version one. During evaluation or prediction:

- dropout layers remain active
- gradients remain disabled
- the model executes `N` stochastic forward passes
- outputs are aggregated into predictive summaries

Classification uncertainty outputs:

- mean probability
- variance of predicted probability
- predictive entropy
- mutual information style epistemic proxy

Difficulty uncertainty outputs:

- mean predicted difficulty
- variance of predicted difficulty

This should be implemented as a separate inference utility rather than folded into the core training step.

## Calibration strategy

Calibration is treated as a post-hoc stage after model training. The repo should support two initial calibration modes:

1. Baseline calibration using the raw classification score only
2. Difficulty-aware calibration using the raw classification score plus predicted difficulty

The first version should keep calibrators lightweight and easy to compare. A practical starting point is logistic calibration on validation-set outputs:

- baseline features: `[logit]`
- difficulty-aware features: `[logit, predicted_difficulty]`

This directly tests the working hypothesis that learned difficulty improves confidence calibration. Temperature scaling can also be included as a strong baseline for score-only calibration.

## Data contract

The first version should support CSV and JSONL datasets with the following fields:

- `text`
- `label`
- `difficulty`
- optional `id`
- optional `split`

If splits are absent, the training script can create them from config. The contract should stay simple and opinionated rather than supporting arbitrary schemas.

## Repository layout

The initial repository shape:

- `pyproject.toml`
- `README.md`
- `configs/`
- `scripts/`
- `src/modernbert_uq/`
- `tests/`
- `docs/plans/`

Proposed Python modules:

- `src/modernbert_uq/config.py`
- `src/modernbert_uq/data.py`
- `src/modernbert_uq/modeling.py`
- `src/modernbert_uq/losses.py`
- `src/modernbert_uq/train.py`
- `src/modernbert_uq/evaluate.py`
- `src/modernbert_uq/uncertainty.py`
- `src/modernbert_uq/calibration.py`
- `src/modernbert_uq/predict.py`
- `src/modernbert_uq/utils.py`

## Configuration

The project should use file-based experiment configuration, with YAML as the first choice. The config must cover:

- pretrained model name
- max sequence length
- optimizer and scheduler settings
- per-device batch sizes
- gradient accumulation
- number of epochs
- mixed precision
- random seed
- loss weights
- dropout settings
- MC-dropout pass count
- dataset paths and split settings
- output directory

The initial code should provide a small default config suitable for smoke tests and local development.

## Metrics

Classification metrics:

- AUROC
- AUPRC
- accuracy
- F1
- log loss / NLL
- Brier score
- ECE

Difficulty metrics:

- RMSE
- MAE
- Pearson or Spearman correlation

Calibration comparison:

- uncalibrated classification performance
- temperature-scaled performance
- difficulty-aware calibration performance

## Testing strategy

Tests should validate the narrow core behavior of the repo:

- model forward shapes for both heads
- loss composition and masking
- tokenization / collation behavior
- MC-dropout stochasticity
- calibrator fitting and prediction
- a minimal single-process training smoke test on synthetic data

Distributed execution can be left to a smoke-level verification path in scripts or manual validation. The automated test suite should stay fast.

## Non-goals for version one

The first version should not include:

- PEFT / LoRA
- DeepSpeed
- FSDP
- active learning loops
- web apps or dashboards
- generic task registries
- support for arbitrary multimodal inputs

## Success criteria

The project is successful when it can:

1. Fine-tune a ModernBERT classifier with classification and difficulty supervision.
2. Run cleanly on a single node with multiple GPUs using `accelerate launch`.
3. Export predictions with MC-dropout uncertainty summaries.
4. Fit and evaluate post-hoc calibrators with and without difficulty features.
5. Provide a reproducible baseline structure for future experiments.
