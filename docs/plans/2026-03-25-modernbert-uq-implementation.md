# ModernBERT UQ Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a fresh `transformers` + `accelerate` repository for ModernBERT-like binary classification with a second difficulty-regression head, MC-dropout uncertainty estimation, and post-hoc calibration that can use predicted difficulty.

**Architecture:** Use a native PyTorch training loop with Accelerate on top of a Hugging Face encoder model. Keep the model surface small and explicit: one shared encoder, one binary head, one regression head, plus separate modules for uncertainty inference and post-hoc calibration.

**Tech Stack:** Python, PyTorch, Transformers, Accelerate, scikit-learn, pytest, YAML config files

**Environment note:** The codebase targets a remote GPU server. The local dev machine does not have PyTorch or GPU support. Tests that require `torch` must be skipped locally using the `requires_torch` marker defined in `conftest.py`. Tests that only use pure Python, YAML, numpy, or scikit-learn must remain runnable locally without any skip markers.

---

### Task 0: Add pytest torch-skip infrastructure

**Files:**
- Create: `tests/conftest.py`

**Step 1: Write implementation**

Create `tests/conftest.py` that defines a `requires_torch` marker and registers a session-scoped fixture to skip tests automatically:

```python
import pytest

def pytest_configure(config):
    config.addinivalue_line(
        "markers", "requires_torch: skip test if torch is not installed"
    )

def pytest_collection_modifyitems(config, items):
    try:
        import torch  # noqa: F401
        torch_available = True
    except ImportError:
        torch_available = False

    skip_torch = pytest.mark.skip(reason="torch not installed")
    for item in items:
        if "requires_torch" in item.keywords and not torch_available:
            item.add_marker(skip_torch)
```

**Step 2: Verify (no torch needed)**

Run: `pytest --collect-only tests/` — should collect without errors on a no-torch machine.

**Step 3: Commit**

```bash
git add tests/conftest.py
git commit -m "test: add requires_torch skip marker"
```

---

### Task 1: Scaffold the Python project

**Files:**
- Create: `pyproject.toml`
- Create: `README.md`
- Create: `src/modernbert_uq/__init__.py`
- Create: `src/modernbert_uq/config.py`
- Create: `configs/default.yaml`
- Test: `tests/test_config.py`

**Torch requirement:** None — config loading is pure Python + YAML.

**Step 1: Write the failing test**

Create `tests/test_config.py` with a test that loads `configs/default.yaml` through `modernbert_uq.config` and asserts key fields like `model_name`, `max_length`, and `train_batch_size` exist. No `requires_torch` marker needed.

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL because the package and config loader do not exist yet.

**Step 3: Write minimal implementation**

Create the package skeleton, add `pyproject.toml` with runtime and dev dependencies, add a typed config loader, and add a default YAML config that captures model, data, optimization, uncertainty, and calibration settings.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py -v`
Expected: PASS (runs locally without torch)

**Step 5: Commit**

```bash
git add pyproject.toml README.md src/modernbert_uq/__init__.py src/modernbert_uq/config.py configs/default.yaml tests/test_config.py
git commit -m "feat: scaffold modernbert uq project"
```

---

### Task 2: Implement the dataset and tokenizer pipeline

**Files:**
- Create: `src/modernbert_uq/data.py`
- Test: `tests/test_data.py`

**Torch requirement:** Partial. The tokenizer and collator emit PyTorch tensors, so the full test requires torch. Split the test into:
- A pure-Python section (CSV/JSONL parsing, schema validation) — no marker needed.
- A tokenization/collation section — mark with `@pytest.mark.requires_torch`.

**Step 1: Write the failing test**

Create `tests/test_data.py` with two test functions:

1. `test_data_loading` — no torch marker — verifies a synthetic dataframe/JSONL fixture loads records with `text`, `label`, `difficulty` fields correctly.
2. `test_tokenization_and_collation` — decorated `@pytest.mark.requires_torch` — verifies tokenization respects `max_length` and the collator emits tensors for `input_ids`, `attention_mask`, `labels`, `difficulty`.

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_data.py -v`
Expected: `test_data_loading` FAILS (implementation missing); `test_tokenization_and_collation` SKIPPED locally.

**Step 3: Write minimal implementation**

Implement data loading for CSV and JSONL, split handling, tokenization with a Hugging Face tokenizer, and a collator that preserves both supervision targets. Support optional missing labels through masks.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_data.py -v`
Expected: `test_data_loading` PASS; `test_tokenization_and_collation` SKIPPED locally, PASS on remote.

**Step 5: Commit**

```bash
git add src/modernbert_uq/data.py tests/test_data.py
git commit -m "feat: add dataset and tokenization pipeline"
```

---

### Task 3: Implement the dual-head ModernBERT model

**Files:**
- Create: `src/modernbert_uq/modeling.py`
- Create: `src/modernbert_uq/losses.py`
- Test: `tests/test_modeling.py`

**Torch requirement:** Full — all tests require torch. Mark entire test file with `pytestmark = pytest.mark.requires_torch` at module level.

**Step 1: Write the failing test**

Create `tests/test_modeling.py` with module-level `pytestmark = pytest.mark.requires_torch`. Tests assert:

- classification logits are shaped `[batch]` or `[batch, 1]`
- difficulty predictions are shaped `[batch]`
- masked joint loss returns a scalar tensor

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_modeling.py -v`
Expected: SKIPPED locally (no torch). On remote: FAIL because model and loss functions are missing.

**Step 3: Write minimal implementation**

Implement:

- a shared encoder loaded from `AutoModel`
- a pooling path using the encoder output
- one dropout layer before both heads
- one linear head for classification
- one linear head for difficulty regression with sigmoid squashing to `[0, 1]`
- weighted BCE + MSE loss with optional masks

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_modeling.py -v`
Expected: SKIPPED locally; PASS on remote.

**Step 5: Commit**

```bash
git add src/modernbert_uq/modeling.py src/modernbert_uq/losses.py tests/test_modeling.py
git commit -m "feat: add dual-head modernbert model"
```

---

### Task 4: Build the native Accelerate training loop

**Files:**
- Create: `src/modernbert_uq/train.py`
- Create: `src/modernbert_uq/utils.py`
- Create: `scripts/train.sh`
- Test: `tests/test_train_smoke.py`

**Torch requirement:** Full — training requires torch + accelerate. Mark entire file with `pytestmark = pytest.mark.requires_torch`.

**Step 1: Write the failing test**

Create `tests/test_train_smoke.py` with module-level `pytestmark = pytest.mark.requires_torch`. Tiny synthetic dataset, one-step CPU smoke test verifying:

- model initialization
- dataloader creation
- one optimizer step
- checkpoint or training state artifact saved

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_train_smoke.py -v`
Expected: SKIPPED locally. On remote: FAIL because training loop does not exist.

**Step 3: Write minimal implementation**

Implement a native training loop using `Accelerator` with:

- distributed-aware dataloaders
- optimizer and scheduler creation
- gradient accumulation
- mixed precision support
- checkpoint saving
- periodic validation hooks

Add a shell script showing the standard `accelerate launch` entrypoint for single-node multi-GPU runs.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_train_smoke.py -v`
Expected: SKIPPED locally; PASS on remote.

**Step 5: Commit**

```bash
git add src/modernbert_uq/train.py src/modernbert_uq/utils.py scripts/train.sh tests/test_train_smoke.py
git commit -m "feat: add accelerate training loop"
```

---

### Task 5: Add evaluation metrics and prediction export

**Files:**
- Create: `src/modernbert_uq/evaluate.py`
- Create: `src/modernbert_uq/predict.py`
- Test: `tests/test_evaluate.py`

**Torch requirement:** None for the metric computation tests — use deterministic numpy arrays as inputs (no model forward pass). The `predict.py` module calls the model, but its unit test can use pre-computed arrays. No `requires_torch` marker needed on `test_evaluate.py`.

**Step 1: Write the failing test**

Create `tests/test_evaluate.py` with deterministic numpy arrays for labels, probabilities, and difficulty predictions, then assert the evaluation layer returns classification metrics, regression metrics, and calibration fields. No torch marker needed.

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_evaluate.py -v`
Expected: FAIL (runs locally without torch).

**Step 3: Write minimal implementation**

Implement metric computation for classification and difficulty regression, plus batch prediction export to a dataframe or JSON-serializable structure suitable for calibration fitting. Keep metric functions independent of torch so they accept plain numpy arrays.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_evaluate.py -v`
Expected: PASS locally without torch.

**Step 5: Commit**

```bash
git add src/modernbert_uq/evaluate.py src/modernbert_uq/predict.py tests/test_evaluate.py
git commit -m "feat: add evaluation and prediction export"
```

---

### Task 6: Add MC-dropout uncertainty inference

**Files:**
- Create: `src/modernbert_uq/uncertainty.py`
- Test: `tests/test_uncertainty.py`

**Torch requirement:** Full — stochastic forward passes require torch. Mark entire file with `pytestmark = pytest.mark.requires_torch`.

**Step 1: Write the failing test**

Create `tests/test_uncertainty.py` with module-level `pytestmark = pytest.mark.requires_torch`. Tests assert repeated stochastic forward passes produce non-identical outputs and aggregated uncertainty summaries contain the expected keys.

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_uncertainty.py -v`
Expected: SKIPPED locally. On remote: FAIL because uncertainty utilities are missing.

**Step 3: Write minimal implementation**

Implement utilities that:

- enable dropout during prediction without enabling gradients
- run configurable Monte Carlo passes
- aggregate class probabilities and difficulty predictions
- compute predictive variance, entropy, and epistemic proxy values

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_uncertainty.py -v`
Expected: SKIPPED locally; PASS on remote.

**Step 5: Commit**

```bash
git add src/modernbert_uq/uncertainty.py tests/test_uncertainty.py
git commit -m "feat: add mc-dropout uncertainty inference"
```

---

### Task 7: Add post-hoc calibration with optional difficulty features

**Files:**
- Create: `src/modernbert_uq/calibration.py`
- Test: `tests/test_calibration.py`

**Torch requirement:** None — calibration is pure sklearn + numpy. No `requires_torch` marker needed.

**Step 1: Write the failing test**

Create `tests/test_calibration.py` with synthetic numpy logits, labels, and predicted difficulties, then assert:

- a baseline score-only calibrator can fit and predict probabilities
- a difficulty-aware calibrator can fit on `[logit, difficulty]`
- both produce outputs in `[0, 1]`

No torch marker needed.

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_calibration.py -v`
Expected: FAIL (runs locally without torch).

**Step 3: Write minimal implementation**

Implement:

- temperature scaling for score-only calibration
- a simple logistic-regression calibrator
- a feature-based calibration path that accepts predicted difficulty

Keep the interface small so evaluation can compare uncalibrated, temperature-scaled, and difficulty-aware calibrated probabilities.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_calibration.py -v`
Expected: PASS locally without torch.

**Step 5: Commit**

```bash
git add src/modernbert_uq/calibration.py tests/test_calibration.py
git commit -m "feat: add calibration module"
```

---

### Task 8: Final documentation and integration verification

**Files:**
- Modify: `README.md`
- Create: `scripts/evaluate.sh`
- Create: `scripts/predict_mc_dropout.sh`
- Test: `tests/test_end_to_end_smoke.py`

**Torch requirement:** Full — end-to-end pipeline requires torch. Mark entire file with `pytestmark = pytest.mark.requires_torch`.

**Step 1: Write the failing test**

Create `tests/test_end_to_end_smoke.py` with module-level `pytestmark = pytest.mark.requires_torch` that runs a tiny end-to-end pipeline on synthetic data:

- train for one short pass
- export predictions
- fit a calibrator
- verify artifacts are created

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_end_to_end_smoke.py -v`
Expected: SKIPPED locally. On remote: FAIL because the integrated pipeline is incomplete.

**Step 3: Write minimal implementation**

Document installation, data format, single-node multi-GPU launch, MC-dropout prediction, and calibration workflow in `README.md`. Add small helper scripts for evaluation and uncertainty prediction. Fill integration gaps exposed by the smoke test.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_end_to_end_smoke.py -v`
Expected: SKIPPED locally; PASS on remote.

**Step 5: Commit**

```bash
git add README.md scripts/evaluate.sh scripts/predict_mc_dropout.sh tests/test_end_to_end_smoke.py
git commit -m "docs: finalize usage and smoke coverage"
```

---

### Local vs remote test summary

| Test file | Torch required | Runs locally |
|---|---|---|
| `test_config.py` | No | Yes |
| `test_data.py::test_data_loading` | No | Yes |
| `test_data.py::test_tokenization_and_collation` | Yes | Skipped |
| `test_modeling.py` | Yes | Skipped |
| `test_train_smoke.py` | Yes | Skipped |
| `test_evaluate.py` | No | Yes |
| `test_uncertainty.py` | Yes | Skipped |
| `test_calibration.py` | No | Yes |
| `test_end_to_end_smoke.py` | Yes | Skipped |
