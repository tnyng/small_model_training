# ModernBERT UQ

Focused training code for dual-head ModernBERT classification with difficulty prediction, MC-dropout uncertainty, and post-hoc calibration experiments.

## Data contract

Input datasets should provide:

- `text`
- `label`
- `difficulty`
- optional `id`
- optional `split`

CSV and JSONL inputs are supported by the data utilities.

## Local testing

This repository supports local development on machines without `torch`. Tests that require `torch` are marked with `requires_torch` and skip automatically when the dependency is unavailable.

```bash
pytest tests/test_config.py -v
pytest tests/test_data.py -v
pytest tests/test_evaluate.py -v
pytest tests/test_calibration.py -v
```

## Training and inference

Single-node multi-GPU training is intended to run through `accelerate launch`.

```bash
bash scripts/train.sh
bash scripts/evaluate.sh
bash scripts/predict_mc_dropout.sh
```

The code paths cover:

- dual-head classification and difficulty prediction
- masked joint BCE + MSE loss
- native `Accelerate` training loop
- MC-dropout inference summaries
- score-only and difficulty-aware post-hoc calibration
