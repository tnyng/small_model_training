# Clariden Experiment Suite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Bradley-Terry loss, regression head-ensemble, data-fraction subsampling, and Slurm batch submission to run 20 experiments (4 conditions × 5 data fractions) simultaneously on Clariden GH200 nodes.

**Architecture:** All 4 training conditions share the same `ModernBertUQModel` with `score_task_type="regression"`. The new `config.loss_type` field ("mse" or "bt") selects the training objective at the loop level; `uq_method` selects head structure ("none" or "head_ensemble"). BT training uses a separate `run_bt_training()` that performs two forward passes per batch; validation always uses the single-text regression path for apples-to-apples comparison.

**Tech Stack:** PyTorch, Transformers, Accelerate, pandas, NumPy, pytest, YAML, Slurm/sbatch

---

## File Map

| File | Change |
|---|---|
| `src/modernbert_uq/config.py` | Add `loss_type`, `center_rewards_coeff`, `pair_score_threshold`, `data_fraction` fields |
| `src/modernbert_uq/losses.py` | Add `bradley_terry_loss()` |
| `src/modernbert_uq/data.py` | Add `PairDataset`, `PairBatchCollator`, `load_pair_records`, `tokenize_pair_records` |
| `src/modernbert_uq/modeling.py` | Remove regression+ensemble guard; add `member_scores` branch + `UQModelOutput.member_scores` |
| `src/modernbert_uq/evaluate.py` | Add `pairwise_accuracy()` |
| `src/modernbert_uq/train.py` | Add `--fraction` flag, `build_pair_dataloader()`, `run_bt_training()`, BT dispatch in `run_experiment()` |
| `scripts/train_prepared_rulefollower_branch_cscs_container.sh` | Pass `FRACTION` env var to accelerate launch |
| `scripts/prepare_bt_pairs.py` | **New** — generate pair CSV from regression score CSV, sampling reproducibly when `--max-pairs > 0` |
| `scripts/summarize_experiment_suite.py` | **New** — walk `outputs/{condition}/frac_{f}/` tree, emit comparison CSV |
| `configs/default.yaml` | Add `loss.type`, `loss.center_rewards_coeff`, `data.fraction`, `data.pair_score_threshold` |
| `configs/reg_base.yaml` | **New** — condition 1 config |
| `configs/reg_ensemble.yaml` | **New** — condition 2 config |
| `configs/bt_base.yaml` | **New** — condition 3 config |
| `configs/bt_ensemble.yaml` | **New** — condition 4 config |
| `scripts/submit_all_experiments_cscs.sh` | **New** — submit all 20 Slurm jobs |
| `tests/test_bt_loss.py` | **New** — pure-Python tests for `bradley_terry_loss` |
| `tests/test_bt_data.py` | **New** — pure-Python tests for pair data pipeline |
| `tests/test_regression_ensemble.py` | **New** — torch tests for `head_ensemble` with `score_task_type="regression"` |
| `docs/superpowers/specs/2026-04-13-clariden-experiment-suite-design.md` | **New** — design spec |

---

## Task 1: Extend ExperimentConfig with new fields

**Files:**
- Modify: `src/modernbert_uq/config.py`
- Modify: `configs/default.yaml`

- [ ] **Step 1: Add four new fields to `ExperimentConfig`**

In `src/modernbert_uq/config.py`, add these four fields after `seed: int = 42`:

```python
    loss_type: str = "mse"            # "mse" or "bt"
    center_rewards_coeff: float = 0.0  # BT reward-centering regularisation coeff
    pair_score_threshold: float = 0.1  # min |score_a - score_b| to form a pair
    data_fraction: float = 1.0         # fraction of train split to use (0 < f ≤ 1)
```

- [ ] **Step 2: Read the new fields in `load_config()`**

In `src/modernbert_uq/config.py`, add these four lines inside `load_config()`, after the `seed=...` line:

```python
        loss_type=str(raw_config["loss"].get("type", "mse")),
        center_rewards_coeff=float(raw_config["loss"].get("center_rewards_coeff", 0.0)),
        pair_score_threshold=float(raw_config["data"].get("pair_score_threshold", 0.1)),
        data_fraction=float(raw_config["data"].get("fraction", 1.0)),
```

- [ ] **Step 3: Add defaults to `configs/default.yaml`**

In `configs/default.yaml`, replace the `loss:` block and extend the `data:` block:

```yaml
loss:
  classification_weight: 1.0
  difficulty_weight: 1.0
  type: mse
  center_rewards_coeff: 0.0

data:
  train_path: data/train.csv
  validation_path: data/validation.csv
  test_path: data/test.csv
  fraction: 1.0
  pair_score_threshold: 0.1
```

- [ ] **Step 4: Verify config loads cleanly**

```bash
cd /mnt/c/RuleFollower/modernbert_uq_with_reg_base_data/modernbert_uq
python -c "
from modernbert_uq.config import load_config
c = load_config('configs/default.yaml')
assert c.loss_type == 'mse'
assert c.center_rewards_coeff == 0.0
assert c.pair_score_threshold == 0.1
assert c.data_fraction == 1.0
print('OK')
"
```

Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add src/modernbert_uq/config.py configs/default.yaml
git commit -m "feat: add loss_type, center_rewards_coeff, pair_score_threshold, data_fraction to ExperimentConfig"
```

---

## Task 2: Bradley-Terry loss

**Files:**
- Modify: `src/modernbert_uq/losses.py`
- Create: `tests/test_bt_loss.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_bt_loss.py`:

```python
from __future__ import annotations

import pytest

pytest.importorskip("torch")

import torch

from modernbert_uq.losses import bradley_terry_loss


def test_bt_loss_returns_scalar():
    score_a = torch.tensor([1.0, 0.8])
    score_b = torch.tensor([0.0, 0.2])
    loss = bradley_terry_loss(score_a, score_b)
    assert loss.shape == ()


def test_bt_loss_is_positive():
    score_a = torch.tensor([1.0, 0.8])
    score_b = torch.tensor([0.0, 0.2])
    loss = bradley_terry_loss(score_a, score_b)
    assert float(loss) > 0.0


def test_bt_loss_lower_when_preferred_wins_more():
    # Clear preference → lower loss
    loss_clear = bradley_terry_loss(torch.tensor([2.0]), torch.tensor([-2.0]))
    loss_ambiguous = bradley_terry_loss(torch.tensor([0.1]), torch.tensor([-0.1]))
    assert float(loss_clear) < float(loss_ambiguous)


def test_bt_loss_center_regularization_increases_loss():
    score_a = torch.tensor([1.0])
    score_b = torch.tensor([0.0])
    loss_no_reg = bradley_terry_loss(score_a, score_b, center_rewards_coeff=0.0)
    loss_with_reg = bradley_terry_loss(score_a, score_b, center_rewards_coeff=1.0)
    assert float(loss_with_reg) > float(loss_no_reg)


def test_bt_loss_gradients_flow():
    score_a = torch.tensor([0.5], requires_grad=True)
    score_b = torch.tensor([0.5], requires_grad=True)
    loss = bradley_terry_loss(score_a, score_b)
    loss.backward()
    assert score_a.grad is not None
    assert score_b.grad is not None
    # Preferred (score_a) should have negative gradient (loss decreases as score_a increases)
    assert float(score_a.grad) < 0.0
    # Rejected (score_b) should have positive gradient
    assert float(score_b.grad) > 0.0
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /mnt/c/RuleFollower/modernbert_uq_with_reg_base_data/modernbert_uq
python -m pytest tests/test_bt_loss.py -v 2>&1 | head -20
```

Expected: `ImportError` or `cannot import name 'bradley_terry_loss'`

- [ ] **Step 3: Add `bradley_terry_loss` to `losses.py`**

In `src/modernbert_uq/losses.py`, add after the imports and before `compute_joint_loss`:

```python
def bradley_terry_loss(
    score_a: torch.Tensor,
    score_b: torch.Tensor,
    center_rewards_coeff: float = 0.0,
) -> torch.Tensor:
    """Bradley-Terry pairwise ranking loss.

    score_a is the predicted reward for the preferred (higher-scored) text,
    score_b is the predicted reward for the rejected (lower-scored) text.
    """
    loss = -F.logsigmoid(score_a - score_b).mean()
    if center_rewards_coeff > 0.0:
        loss = loss + center_rewards_coeff * ((score_a + score_b) ** 2).mean()
    return loss
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
python -m pytest tests/test_bt_loss.py -v
```

Expected: all 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/modernbert_uq/losses.py tests/test_bt_loss.py
git commit -m "feat: add bradley_terry_loss to losses.py"
```

---

## Task 3: Pair data pipeline

**Files:**
- Modify: `src/modernbert_uq/data.py`
- Create: `tests/test_bt_data.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_bt_data.py`:

```python
from __future__ import annotations

import io
import textwrap

import pandas as pd
import pytest


def make_pair_csv(tmp_path):
    content = textwrap.dedent("""\
        text_a,text_b,score_a,score_b
        "hello world","goodbye world",0.8,0.2
        "foo bar","baz qux",0.9,0.3
    """)
    path = tmp_path / "train_pairs.csv"
    path.write_text(content)
    return path


def test_load_pair_records(tmp_path):
    from modernbert_uq.data import load_pair_records

    path = make_pair_csv(tmp_path)
    records = load_pair_records(path)
    assert len(records) == 2
    assert set(records[0].keys()) >= {"text_a", "text_b", "score_a", "score_b"}
    assert records[0]["text_a"] == "hello world"
    assert float(records[0]["score_a"]) == pytest.approx(0.8)


def test_load_pair_records_missing_column(tmp_path):
    from modernbert_uq.data import load_pair_records

    bad = tmp_path / "bad.csv"
    bad.write_text("text_a,text_b\nhello,world\n")
    with pytest.raises(ValueError, match="missing columns"):
        load_pair_records(bad)


def test_pair_dataset_len_and_getitem(tmp_path):
    from modernbert_uq.data import PairDataset

    records = [{"a": 1}, {"a": 2}]
    ds = PairDataset(records)
    assert len(ds) == 2
    assert ds[0] == {"a": 1}


def test_pair_batch_collator():
    pytest.importorskip("torch")
    import torch
    from modernbert_uq.data import PairBatchCollator

    features = [
        {
            "input_ids_a": [1, 2, 3],
            "attention_mask_a": [1, 1, 1],
            "input_ids_b": [4, 5, 6],
            "attention_mask_b": [1, 1, 0],
            "score_a": 0.8,
            "score_b": 0.2,
        },
        {
            "input_ids_a": [7, 8, 9],
            "attention_mask_a": [1, 1, 1],
            "input_ids_b": [10, 11, 12],
            "attention_mask_b": [1, 0, 0],
            "score_a": 0.9,
            "score_b": 0.1,
        },
    ]
    collator = PairBatchCollator()
    batch = collator(features)
    assert batch["input_ids_a"].shape == (2, 3)
    assert batch["input_ids_b"].shape == (2, 3)
    assert batch["score_a"].dtype == torch.float32
    assert batch["score_b"].shape == (2,)
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest tests/test_bt_data.py -v 2>&1 | head -20
```

Expected: `ImportError` for `load_pair_records`, `PairDataset`, `PairBatchCollator`

- [ ] **Step 3: Add pair data classes to `data.py`**

In `src/modernbert_uq/data.py`, append after the `_validate_records` function:

```python
def load_pair_records(path: str | Path) -> list[dict[str, Any]]:
    """Load a pair CSV with columns text_a, text_b, score_a, score_b."""
    source_path = Path(path)
    frame = pd.read_csv(source_path)
    required = {"text_a", "text_b", "score_a", "score_b"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Pair CSV missing columns: {sorted(missing)}")
    return frame.to_dict(orient="records")


def tokenize_pair_records(
    records: list[dict[str, Any]],
    tokenizer: Any,
    max_length: int,
) -> list[dict[str, Any]]:
    """Tokenize both texts in each pair record."""
    texts_a = [str(r["text_a"]) for r in records]
    texts_b = [str(r["text_b"]) for r in records]
    tok_a = tokenizer(texts_a, padding="max_length", truncation=True, max_length=max_length)
    tok_b = tokenizer(texts_b, padding="max_length", truncation=True, max_length=max_length)
    output = []
    for i, record in enumerate(records):
        output.append(
            {
                "input_ids_a": tok_a["input_ids"][i],
                "attention_mask_a": tok_a["attention_mask"][i],
                "input_ids_b": tok_b["input_ids"][i],
                "attention_mask_b": tok_b["attention_mask"][i],
                "score_a": float(record["score_a"]),
                "score_b": float(record["score_b"]),
            }
        )
    return output


@dataclass
class PairDataset:
    records: list[dict[str, Any]]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


class PairBatchCollator:
    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        return {
            "input_ids_a": torch.tensor(
                [f["input_ids_a"] for f in features], dtype=torch.long
            ),
            "attention_mask_a": torch.tensor(
                [f["attention_mask_a"] for f in features], dtype=torch.long
            ),
            "input_ids_b": torch.tensor(
                [f["input_ids_b"] for f in features], dtype=torch.long
            ),
            "attention_mask_b": torch.tensor(
                [f["attention_mask_b"] for f in features], dtype=torch.long
            ),
            "score_a": torch.tensor([f["score_a"] for f in features], dtype=torch.float32),
            "score_b": torch.tensor([f["score_b"] for f in features], dtype=torch.float32),
        }
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
python -m pytest tests/test_bt_data.py -v
```

Expected: all 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/modernbert_uq/data.py tests/test_bt_data.py
git commit -m "feat: add PairDataset, PairBatchCollator, load_pair_records, tokenize_pair_records"
```

---

## Task 4: Regression head-ensemble in modeling.py

**Files:**
- Modify: `src/modernbert_uq/modeling.py`
- Create: `tests/test_regression_ensemble.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_regression_ensemble.py`:

```python
from __future__ import annotations

import pytest

pytest.importorskip("torch")

import torch
import torch.nn as nn

from modernbert_uq.modeling import ModernBertUQModel


def _make_tiny_encoder(hidden_size: int = 16) -> nn.Module:
    """Minimal encoder stub: returns last_hidden_state of shape [batch, seq, hidden]."""

    class TinyEncoder(nn.Module):
        class Config:
            hidden_size = 16

        def __init__(self):
            super().__init__()
            self.config = TinyEncoder.Config()
            self.linear = nn.Linear(16, 16)

        def forward(self, input_ids, attention_mask=None):
            batch, seq = input_ids.shape
            hidden = self.linear(torch.zeros(batch, seq, 16))

            class Out:
                last_hidden_state = hidden

            return Out()

    enc = TinyEncoder()
    enc.config.hidden_size = hidden_size
    return enc


@pytest.mark.requires_torch
def test_regression_ensemble_forward_shape():
    encoder = _make_tiny_encoder(hidden_size=16)
    model = ModernBertUQModel(
        encoder=encoder,
        hidden_size=16,
        score_task_type="regression",
        uq_method="head_ensemble",
        num_ensemble_members=3,
    )
    input_ids = torch.zeros(2, 8, dtype=torch.long)
    out = model(input_ids=input_ids)
    assert out.score_pred.shape == (2,)
    assert out.member_scores is not None
    assert out.member_scores.shape == (2, 3)


@pytest.mark.requires_torch
def test_regression_ensemble_member_scores_in_range():
    encoder = _make_tiny_encoder(hidden_size=16)
    model = ModernBertUQModel(
        encoder=encoder,
        hidden_size=16,
        score_task_type="regression",
        uq_method="head_ensemble",
        num_ensemble_members=3,
    )
    input_ids = torch.zeros(4, 8, dtype=torch.long)
    out = model(input_ids=input_ids)
    # After sigmoid all member scores should be in (0, 1)
    assert (out.member_scores > 0.0).all()
    assert (out.member_scores < 1.0).all()


@pytest.mark.requires_torch
def test_regression_ensemble_loss_computed():
    encoder = _make_tiny_encoder(hidden_size=16)
    model = ModernBertUQModel(
        encoder=encoder,
        hidden_size=16,
        score_task_type="regression",
        uq_method="head_ensemble",
        num_ensemble_members=3,
    )
    input_ids = torch.zeros(2, 8, dtype=torch.long)
    labels = torch.tensor([0.2, 0.8])
    label_mask = torch.tensor([True, True])
    out = model(input_ids=input_ids, labels=labels, label_mask=label_mask)
    assert out.loss is not None
    assert out.loss.shape == ()
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest tests/test_regression_ensemble.py -v 2>&1 | head -30
```

Expected: FAIL with `ValueError: head_ensemble is not supported for score regression mode`

- [ ] **Step 3: Add `member_scores` field to `UQModelOutput`**

In `src/modernbert_uq/modeling.py`, add `member_scores` to the `UQModelOutput` dataclass (after `member_logits`):

```python
@dataclass
class UQModelOutput:
    logits_cls: torch.Tensor
    score_pred: torch.Tensor
    difficulty_pred: torch.Tensor
    loss: torch.Tensor | None = None
    pooled_output: torch.Tensor | None = None
    member_logits: torch.Tensor | None = None
    member_scores: torch.Tensor | None = None        # [batch, N] for regression ensemble
    mean_probability: torch.Tensor | None = None
    probability_variance: torch.Tensor | None = None
    predictive_entropy: torch.Tensor | None = None
    mutual_information: torch.Tensor | None = None
    uncertainty_method: str = "none"
```

- [ ] **Step 4: Remove the regression+ensemble guard in `__init__`**

In `src/modernbert_uq/modeling.py`, remove these two lines from `__init__` (currently lines 70-71):

```python
        if self.score_task_type == "regression" and uq_method == "head_ensemble":
            raise ValueError("head_ensemble is not supported for score regression mode")
```

- [ ] **Step 5: Add regression ensemble branch in `forward()`**

In `src/modernbert_uq/modeling.py`, replace the entire `if self.uq_method == "head_ensemble":` block in `forward()` with this expanded version:

```python
        member_logits = None
        member_scores = None
        mean_probability = None
        probability_variance = None
        predictive_entropy = None
        mutual_information = None

        if self.uq_method == "head_ensemble" and self.score_task_type == "regression":
            # Regression ensemble: N independent heads → sigmoid → [batch, N]
            member_raw = torch.stack(
                [head(pooled).squeeze(-1) for head in self.classification_head],
                dim=1,
            )  # [batch, N]
            member_scores = torch.sigmoid(member_raw)  # [batch, N]
            logits_cls = member_raw.mean(dim=1)        # [batch]
            score_pred = member_scores.mean(dim=1)     # [batch]
        elif self.uq_method == "head_ensemble":
            # Classification ensemble (existing behaviour)
            member_logits = torch.stack(
                [head(pooled).squeeze(-1) for head in self.classification_head],
                dim=1,
            )
            logits_cls = member_logits.mean(dim=1)
            probability_samples = torch.sigmoid(member_logits)
            mean_probability = probability_samples.mean(dim=1)
            probability_variance = probability_samples.var(dim=1, unbiased=False)
            sample_entropy = binary_entropy(probability_samples)
            predictive_entropy = binary_entropy(mean_probability)
            mutual_information = predictive_entropy - sample_entropy.mean(dim=1)
            score_pred = mean_probability
        else:
            logits_cls = self.classification_head(pooled).squeeze(-1)
            score_pred = mean_probability if mean_probability is not None else torch.sigmoid(logits_cls)
```

- [ ] **Step 6: Add regression ensemble loss computation**

In `src/modernbert_uq/modeling.py`, replace the loss computation block (the existing `if self.uq_method == "head_ensemble":` loss section) with:

```python
        if self.uq_method == "head_ensemble" and self.score_task_type == "regression":
            # Average MSE across all ensemble members
            difficulty_pred = torch.sigmoid(self.difficulty_head(pooled).squeeze(-1))
            member_losses = []
            if labels is not None and label_mask is not None:
                lm = label_mask.to(dtype=torch.bool, device=member_scores.device)
                if torch.any(lm):
                    lab = labels.to(device=member_scores.device, dtype=member_scores.dtype)
                    for k in range(self.num_ensemble_members):
                        import torch.nn.functional as F as _F
                        member_losses.append(_F.mse_loss(member_scores[lm, k], lab[lm]))
            score_loss = torch.stack(member_losses).mean() if member_losses else None
            difficulty_loss = compute_joint_loss(
                logits_cls=logits_cls,
                score_pred=score_pred,
                difficulty_pred=difficulty_pred,
                labels=None,
                difficulty_targets=difficulty,
                label_mask=None,
                difficulty_mask=difficulty_mask,
                score_task_type=self.score_task_type,
                cls_weight=0.0,
                difficulty_weight=self.difficulty_loss_weight,
                positive_class_weight=self.positive_class_weight,
                negative_class_weight=self.negative_class_weight,
            )
            loss = None
            if score_loss is not None:
                loss = self.score_loss_weight * score_loss
            if difficulty_loss is not None:
                loss = difficulty_loss if loss is None else loss + difficulty_loss
        elif self.uq_method == "head_ensemble":
            difficulty_pred = torch.sigmoid(self.difficulty_head(pooled).squeeze(-1))
            cls_loss = compute_ensemble_classification_loss(
                member_logits=member_logits,
                labels=labels,
                label_mask=label_mask,
            )
            difficulty_loss = compute_joint_loss(
                logits_cls=logits_cls,
                score_pred=score_pred,
                difficulty_pred=difficulty_pred,
                labels=None,
                difficulty_targets=difficulty,
                label_mask=None,
                difficulty_mask=difficulty_mask,
                score_task_type=self.score_task_type,
                cls_weight=0.0,
                difficulty_weight=self.difficulty_loss_weight,
                positive_class_weight=self.positive_class_weight,
                negative_class_weight=self.negative_class_weight,
            )
            loss = None
            if cls_loss is not None:
                loss = self.score_loss_weight * cls_loss
            if difficulty_loss is not None:
                loss = difficulty_loss if loss is None else loss + difficulty_loss
        else:
            difficulty_pred = torch.sigmoid(self.difficulty_head(pooled).squeeze(-1))
            loss = compute_joint_loss(
                logits_cls=logits_cls,
                score_pred=score_pred,
                difficulty_pred=difficulty_pred,
                labels=labels,
                difficulty_targets=difficulty,
                label_mask=label_mask,
                difficulty_mask=difficulty_mask,
                score_task_type=self.score_task_type,
                cls_weight=self.score_loss_weight,
                difficulty_weight=self.difficulty_loss_weight,
                positive_class_weight=self.positive_class_weight,
                negative_class_weight=self.negative_class_weight,
            )
```

> **Note:** The `import torch.nn.functional as F as _F` is syntactically wrong. Use `import torch.nn.functional as _F` on its own line inside the block, or simply reference the already-imported `F` since `losses.py` imports it. In `modeling.py`, `F` is already imported at the top (`import torch.nn.functional as F`), so just use `F.mse_loss` directly.

- [ ] **Step 7: Fix the F import reference**

Replace the `import torch.nn.functional as F as _F` line in the block above with:

```python
                    member_losses.append(F.mse_loss(member_scores[lm, k], lab[lm]))
```

(Remove the bad import line; `F` is already imported at the top of `modeling.py`.)

- [ ] **Step 8: Add `member_scores` to the return value**

In the `return UQModelOutput(...)` at the bottom of `forward()`, add `member_scores=member_scores,`:

```python
        return UQModelOutput(
            logits_cls=logits_cls,
            score_pred=score_pred,
            difficulty_pred=difficulty_pred,
            loss=loss,
            pooled_output=pooled,
            member_logits=member_logits,
            member_scores=member_scores,
            mean_probability=mean_probability,
            probability_variance=probability_variance,
            predictive_entropy=predictive_entropy,
            mutual_information=mutual_information,
            uncertainty_method=self.uq_method,
        )
```

- [ ] **Step 9: Run tests to confirm they pass**

```bash
python -m pytest tests/test_regression_ensemble.py -v
```

Expected: all 3 tests PASS (or SKIP if torch not installed locally)

- [ ] **Step 10: Commit**

```bash
git add src/modernbert_uq/modeling.py tests/test_regression_ensemble.py
git commit -m "feat: support head_ensemble for regression task type in ModernBertUQModel"
```

---

## Task 5: `--fraction` flag for training data subsampling

**Files:**
- Modify: `src/modernbert_uq/train.py`

- [ ] **Step 1: Add `--fraction` to `parse_args()`**

In `src/modernbert_uq/train.py`, replace `parse_args()`:

```python
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ModernBERT UQ from a YAML config.")
    parser.add_argument("--config", type=Path, required=True, help="Path to experiment config YAML.")
    parser.add_argument(
        "--fraction",
        type=float,
        default=None,
        help="Fraction of training data to use [0, 1]. Overrides config data.fraction.",
    )
    return parser.parse_args()
```

- [ ] **Step 2: Add fraction subsampling to `build_dataloader()`**

In `src/modernbert_uq/train.py`, add a `fraction: float = 1.0` parameter to `build_dataloader()` and subsample before tokenizing:

```python
def build_dataloader(
    path: str | Path | None,
    tokenizer,
    max_length: int,
    batch_size: int,
    shuffle: bool,
    target_column: str,
    fraction: float = 1.0,
    seed: int = 42,
) -> tuple[DataLoader | None, list[dict[str, object]]]:
    if not path:
        return None, []
    records = load_records(path, target_column=target_column)
    if fraction < 1.0:
        rng = np.random.default_rng(seed)
        n = max(1, int(len(records) * fraction))
        indices = sorted(rng.choice(len(records), size=n, replace=False).tolist())
        records = [records[i] for i in indices]
    tokenized = tokenize_records(
        records,
        tokenizer=tokenizer,
        max_length=max_length,
        target_column=target_column,
    )
    dataset = TextDifficultyDataset(tokenized)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=UQBatchCollator(),
    )
    return dataloader, records
```

- [ ] **Step 3: Apply fraction and CLI override in `run_experiment()` and `main()`**

In `run_experiment()`, change the `train_loader` call to pass `fraction` and `seed`:

```python
    data_fraction = getattr(config, "data_fraction", 1.0)
    train_loader, _train_records = build_dataloader(
        config.train_path,
        tokenizer=tokenizer,
        max_length=config.max_length,
        batch_size=config.train_batch_size,
        shuffle=True,
        target_column=config.score_target_column,
        fraction=data_fraction,
        seed=config.seed,
    )
```

In `main()`, apply the CLI override before calling `run_experiment`:

```python
def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.fraction is not None:
        import dataclasses
        config = dataclasses.replace(config, data_fraction=args.fraction)
    run_experiment(config)
```

- [ ] **Step 4: Verify manually (no torch required)**

```bash
python -c "
from modernbert_uq.train import parse_args
import sys
sys.argv = ['train', '--config', 'configs/default.yaml', '--fraction', '0.5']
args = parse_args()
assert args.fraction == 0.5
print('OK')
"
```

Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add src/modernbert_uq/train.py
git commit -m "feat: add --fraction flag for training data subsampling"
```

---

## Task 6: BT training branch

**Files:**
- Modify: `src/modernbert_uq/train.py`

- [ ] **Step 1: Add `build_pair_dataloader()` to `train.py`**

In `src/modernbert_uq/train.py`, add this import at the top (inside the existing import block):

```python
from modernbert_uq.data import (
    TextDifficultyDataset,
    UQBatchCollator,
    PairDataset,
    PairBatchCollator,
    load_pair_records,
    tokenize_pair_records,
    load_records,
    tokenize_records,
)
```

Then add `build_pair_dataloader()` after `build_dataloader()`:

```python
def build_pair_dataloader(
    path: str | Path | None,
    tokenizer,
    max_length: int,
    batch_size: int,
    shuffle: bool,
    fraction: float = 1.0,
    seed: int = 42,
) -> DataLoader | None:
    if not path:
        return None
    records = load_pair_records(path)
    if fraction < 1.0:
        rng = np.random.default_rng(seed)
        n = max(1, int(len(records) * fraction))
        indices = sorted(rng.choice(len(records), size=n, replace=False).tolist())
        records = [records[i] for i in indices]
    tokenized = tokenize_pair_records(records, tokenizer=tokenizer, max_length=max_length)
    dataset = PairDataset(tokenized)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=PairBatchCollator(),
    )
```

- [ ] **Step 2: Add `run_bt_training()` to `train.py`**

Add after `run_training()`:

```python
def run_bt_training(
    model: torch.nn.Module,
    train_pair_batches: Iterable[dict[str, torch.Tensor]],
    eval_batches: Iterable[dict[str, torch.Tensor]] | None,
    config: TrainConfig,
    center_rewards_coeff: float = 0.0,
) -> TrainingResult:
    """Training loop for Bradley-Terry objective using pair batches."""
    from modernbert_uq.losses import bradley_terry_loss

    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision,
    )
    output_dir = ensure_directory(config.output_dir)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    model, optimizer = accelerator.prepare(model, optimizer)

    global_step = 0
    best_eval_loss = None
    train_pair_batches = list(train_pair_batches)
    eval_batches = list(eval_batches or [])

    for _ in range(config.num_epochs):
        model.train()
        for batch in train_pair_batches:
            batch = move_batch_to_device(batch, accelerator.device)
            with accelerator.accumulate(model):
                out_a = model(
                    input_ids=batch["input_ids_a"],
                    attention_mask=batch["attention_mask_a"],
                )
                out_b = model(
                    input_ids=batch["input_ids_b"],
                    attention_mask=batch["attention_mask_b"],
                )
                uq_method = getattr(accelerator.unwrap_model(model), "uq_method", "none")
                if uq_method == "head_ensemble" and out_a.member_scores is not None:
                    # Average BT loss across ensemble members
                    n_members = out_a.member_scores.shape[1]
                    bt_losses = [
                        bradley_terry_loss(
                            out_a.member_scores[:, k],
                            out_b.member_scores[:, k],
                            center_rewards_coeff=center_rewards_coeff,
                        )
                        for k in range(n_members)
                    ]
                    loss = torch.stack(bt_losses).mean()
                else:
                    loss = bradley_terry_loss(
                        out_a.score_pred,
                        out_b.score_pred,
                        center_rewards_coeff=center_rewards_coeff,
                    )
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
            global_step += 1

        if eval_batches:
            # Use regression eval loss (single-text MSE) for convergence monitoring
            eval_loss = evaluate_loss(model, eval_batches, accelerator.device)
            if best_eval_loss is None or eval_loss < best_eval_loss:
                best_eval_loss = eval_loss

    artifact_path = output_dir / "training_state.pt"
    unwrapped = accelerator.unwrap_model(model)
    accelerator.save(
        {
            "model_state_dict": unwrapped.state_dict(),
            "global_step": global_step,
            "best_eval_loss": best_eval_loss,
        },
        artifact_path,
    )
    return TrainingResult(
        global_step=global_step,
        best_eval_loss=best_eval_loss,
        artifact_path=artifact_path,
    )
```

- [ ] **Step 3: Dispatch to `run_bt_training()` in `run_experiment()`**

In `run_experiment()`, replace the single `result = run_training(...)` call with this dispatch:

```python
    loss_type = getattr(config, "loss_type", "mse")
    center_rewards_coeff = getattr(config, "center_rewards_coeff", 0.0)
    pair_score_threshold = getattr(config, "pair_score_threshold", 0.1)

    if loss_type == "bt":
        # For BT training, train_path points to a pair CSV (train_pairs.csv)
        pair_train_loader = build_pair_dataloader(
            config.train_path,
            tokenizer=tokenizer,
            max_length=config.max_length,
            batch_size=config.train_batch_size,
            shuffle=True,
            fraction=data_fraction,
            seed=config.seed,
        )
        if pair_train_loader is None:
            raise ValueError("BT training requires train_path pointing to a pair CSV")
        result = run_bt_training(
            model=model,
            train_pair_batches=list(pair_train_loader),
            eval_batches=list(validation_loader) if validation_loader is not None else None,
            config=TrainConfig(
                learning_rate=config.learning_rate,
                num_epochs=config.num_epochs,
                output_dir=output_dir,
                gradient_accumulation_steps=config.gradient_accumulation_steps,
                mixed_precision=config.mixed_precision,
                weight_decay=config.weight_decay,
            ),
            center_rewards_coeff=center_rewards_coeff,
        )
    else:
        result = run_training(
            model=model,
            train_batches=list(train_loader),
            eval_batches=list(validation_loader) if validation_loader is not None else None,
            config=TrainConfig(
                learning_rate=config.learning_rate,
                num_epochs=config.num_epochs,
                output_dir=output_dir,
                gradient_accumulation_steps=config.gradient_accumulation_steps,
                mixed_precision=config.mixed_precision,
                weight_decay=config.weight_decay,
            ),
        )
```

- [ ] **Step 4: Verify the module still imports cleanly**

```bash
python -c "from modernbert_uq.train import run_bt_training, build_pair_dataloader; print('OK')"
```

Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add src/modernbert_uq/train.py
git commit -m "feat: add run_bt_training, build_pair_dataloader, BT dispatch in run_experiment"
```

---

## Task 7: Pairwise accuracy in evaluate.py

**Files:**
- Modify: `src/modernbert_uq/evaluate.py`

- [ ] **Step 1: Write failing test**

In `tests/test_evaluate.py` (existing file), add at the bottom:

```python
def test_pairwise_accuracy_perfect():
    from modernbert_uq.evaluate import pairwise_accuracy
    scores = np.array([0.9, 0.1, 0.8, 0.2])
    # pairs: (0,1) preferred=0, (2,3) preferred=2 — model scores agree
    pairs = [(0, 1), (2, 3)]  # (preferred_idx, rejected_idx)
    acc = pairwise_accuracy(scores, pairs)
    assert acc == pytest.approx(1.0)


def test_pairwise_accuracy_chance():
    from modernbert_uq.evaluate import pairwise_accuracy
    scores = np.array([0.5, 0.5])
    pairs = [(0, 1)]
    acc = pairwise_accuracy(scores, pairs)
    # Equal scores → 0.5 accuracy (tie counted as 0.5)
    assert acc == pytest.approx(0.5)


def test_pairwise_accuracy_empty():
    from modernbert_uq.evaluate import pairwise_accuracy
    acc = pairwise_accuracy(np.array([0.8, 0.2]), [])
    assert np.isnan(acc)
```

- [ ] **Step 2: Run to confirm they fail**

```bash
python -m pytest tests/test_evaluate.py -v -k "pairwise" 2>&1 | head -20
```

Expected: `ImportError` for `pairwise_accuracy`

- [ ] **Step 3: Add `pairwise_accuracy()` to `evaluate.py`**

In `src/modernbert_uq/evaluate.py`, add after `compute_spearman`:

```python
def pairwise_accuracy(
    scores: np.ndarray,
    pairs: list[tuple[int, int]],
) -> float:
    """Fraction of pairs where the preferred item has a higher predicted score.

    Args:
        scores: 1-D array of predicted scores, one per example.
        pairs: list of (preferred_idx, rejected_idx) index pairs.

    Returns:
        Accuracy in [0, 1]; ties count as 0.5. Returns NaN for empty pairs.
    """
    if not pairs:
        return float("nan")
    scores = np.asarray(scores, dtype=float)
    correct = 0.0
    for preferred_idx, rejected_idx in pairs:
        diff = scores[preferred_idx] - scores[rejected_idx]
        if diff > 0:
            correct += 1.0
        elif diff == 0:
            correct += 0.5
    return float(correct / len(pairs))
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
python -m pytest tests/test_evaluate.py -v -k "pairwise"
```

Expected: all 3 new tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/modernbert_uq/evaluate.py tests/test_evaluate.py
git commit -m "feat: add pairwise_accuracy to evaluate.py"
```

---

## Task 8: `prepare_bt_pairs.py` script

**Files:**
- Create: `scripts/prepare_bt_pairs.py`

- [ ] **Step 1: Create the script**

Create `scripts/prepare_bt_pairs.py`:

```python
#!/usr/bin/env python3
"""Generate a Bradley-Terry pair CSV from a regression score CSV.

Reads train.csv from --input-dir, generates all unique pairs (i, j) where
|score_i - score_j| >= --threshold (with score_i > score_j), shuffles, and
writes train_pairs.csv to --output-dir. Copies validation.csv and test.csv unchanged.

Usage:
    python scripts/prepare_bt_pairs.py \
        --input-dir data/rulefollower_heuristic_ann_gpt41mini_reg_base \
        --output-dir data/rulefollower_heuristic_ann_gpt41mini_bt_base \
        --threshold 0.1
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate BT pair CSV from regression CSV.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="Minimum |score_a - score_b| to include a pair (default: 0.1)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--score-column",
        type=str,
        default="score_target",
        help="Column name for regression score (default: score_target)",
    )
    return parser.parse_args()


def generate_pairs(
    frame: pd.DataFrame,
    score_column: str,
    threshold: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    texts = frame["text"].tolist()
    scores = frame[score_column].tolist()
    n = len(texts)
    rows = []
    for i in range(n):
        for j in range(i + 1, n):
            diff = scores[i] - scores[j]
            if abs(diff) >= threshold:
                if diff > 0:
                    preferred_i, rejected_j = i, j
                else:
                    preferred_i, rejected_j = j, i
                rows.append(
                    {
                        "text_a": texts[preferred_i],
                        "text_b": texts[rejected_j],
                        "score_a": scores[preferred_i],
                        "score_b": scores[rejected_j],
                    }
                )
    if not rows:
        raise ValueError(
            f"No pairs found with |score_a - score_b| >= {threshold}. "
            "Try a lower threshold."
        )
    shuffled = rng.permutation(len(rows))
    return pd.DataFrame([rows[k] for k in shuffled])


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_path = args.input_dir / "train.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"train.csv not found in {args.input_dir}")

    train_frame = pd.read_csv(train_path)
    print(f"[INFO] Loaded {len(train_frame)} training rows from {train_path}")

    pairs = generate_pairs(train_frame, args.score_column, args.threshold, rng)
    out_path = args.output_dir / "train_pairs.csv"
    pairs.to_csv(out_path, index=False)
    print(f"[INFO] Wrote {len(pairs)} pairs to {out_path}")

    score_diffs = (pairs["score_a"] - pairs["score_b"]).describe()
    print(f"[INFO] Score-diff distribution:\n{score_diffs}")

    for split in ("val.csv", "test.csv", "validation.csv"):
        src = args.input_dir / split
        if src.exists():
            dst = args.output_dir / split
            shutil.copy2(src, dst)
            print(f"[INFO] Copied {src.name} → {dst}")

    # Write a summary JSON
    import json
    summary = {
        "input_dir": str(args.input_dir),
        "threshold": args.threshold,
        "train_rows_in": int(len(train_frame)),
        "pairs_generated": int(len(pairs)),
        "seed": args.seed,
    }
    (args.output_dir / "pair_summary.json").write_text(json.dumps(summary, indent=2))
    print("[DONE] Pair dataset ready.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Test the script on real data**

```bash
cd /mnt/c/RuleFollower/modernbert_uq_with_reg_base_data/modernbert_uq
python scripts/prepare_bt_pairs.py \
  --input-dir data/rulefollower_heuristic_ann_gpt41mini_reg_base \
  --output-dir data/rulefollower_heuristic_ann_gpt41mini_bt_base \
  --threshold 0.1
```

Expected output: prints pair count and score-diff distribution; creates `data/rulefollower_heuristic_ann_gpt41mini_bt_base/train_pairs.csv`

- [ ] **Step 3: Verify the pair CSV**

```bash
python -c "
import pandas as pd
df = pd.read_csv('data/rulefollower_heuristic_ann_gpt41mini_bt_base/train_pairs.csv')
print('Columns:', list(df.columns))
print('Rows:', len(df))
assert set(df.columns) >= {'text_a','text_b','score_a','score_b'}
assert (df.score_a - df.score_b).min() >= 0.1 - 1e-9
print('OK')
"
```

Expected: prints row count; `OK`

- [ ] **Step 4: Commit**

```bash
git add scripts/prepare_bt_pairs.py data/rulefollower_heuristic_ann_gpt41mini_bt_base/
git commit -m "feat: add prepare_bt_pairs.py; generate BT pair dataset from reg_base"
```

---

## Task 9: Four experiment configs

**Files:**
- Create: `configs/reg_base.yaml`
- Create: `configs/reg_ensemble.yaml`
- Create: `configs/bt_base.yaml`
- Create: `configs/bt_ensemble.yaml`

- [ ] **Step 1: Create `configs/reg_base.yaml`**

```yaml
# Condition 1: regression baseline, single head, MC-dropout UQ
model:
  name: answerdotai/ModernBERT-base
  score_task_type: regression
  max_length: 256
  dropout_prob: 0.1
  pooling_strategy: mean

data:
  train_path: data/rulefollower_heuristic_ann_gpt41mini_reg_base/train.csv
  validation_path: data/rulefollower_heuristic_ann_gpt41mini_reg_base/validation.csv
  test_path: data/rulefollower_heuristic_ann_gpt41mini_reg_base/test.csv
  score_target_column: score_target
  fraction: 1.0
  pair_score_threshold: 0.1

training:
  train_batch_size: 32
  eval_batch_size: 64
  num_epochs: 5
  gradient_accumulation_steps: 1
  mixed_precision: bf16
  seed: 42

optimizer:
  learning_rate: 2.0e-5
  weight_decay: 0.01
  warmup_ratio: 0.1

loss:
  type: mse
  score_weight: 1.0
  difficulty_weight: 0.5
  center_rewards_coeff: 0.0

uncertainty:
  method: none
  num_ensemble_members: 1
  mc_dropout_passes: 16

output:
  dir: outputs/reg_base/frac_1.00
```

- [ ] **Step 2: Create `configs/reg_ensemble.yaml`**

```yaml
# Condition 2: regression + multi-head ensemble UQ
model:
  name: answerdotai/ModernBERT-base
  score_task_type: regression
  max_length: 256
  dropout_prob: 0.1
  pooling_strategy: mean

data:
  train_path: data/rulefollower_heuristic_ann_gpt41mini_reg_base/train.csv
  validation_path: data/rulefollower_heuristic_ann_gpt41mini_reg_base/validation.csv
  test_path: data/rulefollower_heuristic_ann_gpt41mini_reg_base/test.csv
  score_target_column: score_target
  fraction: 1.0
  pair_score_threshold: 0.1

training:
  train_batch_size: 32
  eval_batch_size: 64
  num_epochs: 5
  gradient_accumulation_steps: 1
  mixed_precision: bf16
  seed: 42

optimizer:
  learning_rate: 2.0e-5
  weight_decay: 0.01
  warmup_ratio: 0.1

loss:
  type: mse
  score_weight: 1.0
  difficulty_weight: 0.5
  center_rewards_coeff: 0.0

uncertainty:
  method: head_ensemble
  num_ensemble_members: 5
  mc_dropout_passes: 0

output:
  dir: outputs/reg_ensemble/frac_1.00
```

- [ ] **Step 3: Create `configs/bt_base.yaml`**

```yaml
# Condition 3: Bradley-Terry objective, single head, MC-dropout UQ
model:
  name: answerdotai/ModernBERT-base
  score_task_type: regression
  max_length: 256
  dropout_prob: 0.1
  pooling_strategy: mean

data:
  train_path: data/rulefollower_heuristic_ann_gpt41mini_bt_base/train_pairs.csv
  validation_path: data/rulefollower_heuristic_ann_gpt41mini_bt_base/validation.csv
  test_path: data/rulefollower_heuristic_ann_gpt41mini_bt_base/test.csv
  score_target_column: score_target
  fraction: 1.0
  pair_score_threshold: 0.1

training:
  train_batch_size: 16
  eval_batch_size: 64
  num_epochs: 5
  gradient_accumulation_steps: 2
  mixed_precision: bf16
  seed: 42

optimizer:
  learning_rate: 2.0e-5
  weight_decay: 0.01
  warmup_ratio: 0.1

loss:
  type: bt
  score_weight: 1.0
  difficulty_weight: 0.5
  center_rewards_coeff: 0.01

uncertainty:
  method: none
  num_ensemble_members: 1
  mc_dropout_passes: 16

output:
  dir: outputs/bt_base/frac_1.00
```

- [ ] **Step 4: Create `configs/bt_ensemble.yaml`**

```yaml
# Condition 4: Bradley-Terry objective + multi-head ensemble UQ
model:
  name: answerdotai/ModernBERT-base
  score_task_type: regression
  max_length: 256
  dropout_prob: 0.1
  pooling_strategy: mean

data:
  train_path: data/rulefollower_heuristic_ann_gpt41mini_bt_base/train_pairs.csv
  validation_path: data/rulefollower_heuristic_ann_gpt41mini_bt_base/validation.csv
  test_path: data/rulefollower_heuristic_ann_gpt41mini_bt_base/test.csv
  score_target_column: score_target
  fraction: 1.0
  pair_score_threshold: 0.1

training:
  train_batch_size: 16
  eval_batch_size: 64
  num_epochs: 5
  gradient_accumulation_steps: 2
  mixed_precision: bf16
  seed: 42

optimizer:
  learning_rate: 2.0e-5
  weight_decay: 0.01
  warmup_ratio: 0.1

loss:
  type: bt
  score_weight: 1.0
  difficulty_weight: 0.5
  center_rewards_coeff: 0.01

uncertainty:
  method: head_ensemble
  num_ensemble_members: 5
  mc_dropout_passes: 0

output:
  dir: outputs/bt_ensemble/frac_1.00
```

- [ ] **Step 5: Verify all four configs load**

```bash
python -c "
from modernbert_uq.config import load_config
for name in ['reg_base', 'reg_ensemble', 'bt_base', 'bt_ensemble']:
    c = load_config(f'configs/{name}.yaml')
    print(f'{name}: loss_type={c.loss_type} uq={c.uq_method} members={c.num_ensemble_members}')
"
```

Expected:
```
reg_base: loss_type=mse uq=none members=1
reg_ensemble: loss_type=mse uq=head_ensemble members=5
bt_base: loss_type=bt uq=none members=1
bt_ensemble: loss_type=bt uq=head_ensemble members=5
```

- [ ] **Step 6: Commit**

```bash
git add configs/reg_base.yaml configs/reg_ensemble.yaml configs/bt_base.yaml configs/bt_ensemble.yaml
git commit -m "feat: add four experiment base configs (reg_base, reg_ensemble, bt_base, bt_ensemble)"
```

---

## Task 10: Pass FRACTION through the container Slurm script

**Files:**
- Modify: `scripts/train_prepared_rulefollower_branch_cscs_container.sh`

- [ ] **Step 1: Add `FRACTION` env var and pass it to accelerate launch**

In `scripts/train_prepared_rulefollower_branch_cscs_container.sh`:

After the line `JOB_LABEL="${JOB_LABEL:-$DATASET_DIR}"`, add:

```bash
FRACTION="${FRACTION:-1.0}"
export FRACTION
```

Replace the `TRAIN_CMD` block with:

```bash
TRAIN_CMD="
cd $TRAIN_ROOT
export PYTHONPATH=\"$TRAIN_ROOT/src:\${PYTHONPATH:-}\"
\"\$VENV_DIR/bin/python\" -m accelerate.commands.launch -m modernbert_uq.train \
  --config configs/$CONFIG_NAME \
  --fraction \${FRACTION}
"
```

Also add to the `echo` info block:

```bash
echo "[INFO] FRACTION=$FRACTION"
```

- [ ] **Step 2: Verify the script still passes shellcheck basic syntax**

```bash
bash -n scripts/train_prepared_rulefollower_branch_cscs_container.sh
```

Expected: no output (no syntax errors)

- [ ] **Step 3: Commit**

```bash
git add scripts/train_prepared_rulefollower_branch_cscs_container.sh
git commit -m "feat: pass FRACTION env var to accelerate launch in container script"
```

---

## Task 11: Batch submission script

**Files:**
- Create: `scripts/submit_all_experiments_cscs.sh`

- [ ] **Step 1: Create the script**

Create `scripts/submit_all_experiments_cscs.sh`:

```bash
#!/usr/bin/env bash
# Submit all 20 ModernBERT UQ experiments to Clariden.
# 4 conditions × 5 data fractions = 20 Slurm jobs.
# Each job requests 1 GH200; Slurm packs up to 4 per node automatically.
#
# Usage (from TRAIN_ROOT):
#   bash scripts/submit_all_experiments_cscs.sh
#
# Override container environment:
#   CSCS_ENVIRONMENT=myenv bash scripts/submit_all_experiments_cscs.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SCRIPT="scripts/train_prepared_rulefollower_branch_cscs_container.sh"

if [ ! -f "$SCRIPT" ]; then
  echo "[ERROR] Script not found: $SCRIPT"
  exit 1
fi

declare -A DATASET_MAP
DATASET_MAP[reg_base]="rulefollower_heuristic_ann_gpt41mini_reg_base"
DATASET_MAP[reg_ensemble]="rulefollower_heuristic_ann_gpt41mini_reg_base"
DATASET_MAP[bt_base]="rulefollower_heuristic_ann_gpt41mini_bt_base"
DATASET_MAP[bt_ensemble]="rulefollower_heuristic_ann_gpt41mini_bt_base"

# For BT conditions, train.csv is train_pairs.csv — validate that file exists
declare -A TRAIN_FILE_MAP
TRAIN_FILE_MAP[reg_base]="train.csv"
TRAIN_FILE_MAP[reg_ensemble]="train.csv"
TRAIN_FILE_MAP[bt_base]="train_pairs.csv"
TRAIN_FILE_MAP[bt_ensemble]="train_pairs.csv"

FRACTIONS=(0.1 0.25 0.5 0.75 1.0)
CONDITIONS=(reg_base reg_ensemble bt_base bt_ensemble)

for condition in "${CONDITIONS[@]}"; do
  dataset="${DATASET_MAP[$condition]}"
  train_file="${TRAIN_FILE_MAP[$condition]}"
  if [ ! -f "data/$dataset/$train_file" ]; then
    echo "[ERROR] Missing: data/$dataset/$train_file  — run prepare_bt_pairs.py first for BT conditions"
    exit 1
  fi
  for fraction in "${FRACTIONS[@]}"; do
    frac_label="${fraction/./}"  # e.g. "0.1" → "01"
    job_name="${condition}_f${frac_label}"
    echo "[SBATCH] condition=$condition  fraction=$fraction  dataset=$dataset"
    sbatch \
      --job-name="$job_name" \
      --export=ALL,CONFIG_NAME="${condition}.yaml",DATASET_DIR="$dataset",FRACTION="$fraction",JOB_LABEL="$job_name" \
      "$SCRIPT"
  done
done

echo ""
echo "[DONE] Submitted ${#CONDITIONS[@]} conditions × ${#FRACTIONS[@]} fractions = $((${#CONDITIONS[@]} * ${#FRACTIONS[@]})) jobs."
echo "       Monitor with: squeue --me"
```

- [ ] **Step 2: Make executable and verify syntax**

```bash
chmod +x scripts/submit_all_experiments_cscs.sh
bash -n scripts/submit_all_experiments_cscs.sh
```

Expected: no output (no syntax errors)

- [ ] **Step 3: Dry-run validation (no actual submission)**

```bash
# Temporarily replace sbatch with echo to preview commands
bash -c '
  sbatch() { echo "SBATCH: $*"; }
  export -f sbatch
  bash scripts/submit_all_experiments_cscs.sh
' 2>&1 | head -50
```

Expected: 20 `SBATCH:` lines, one per experiment, with correct `CONFIG_NAME`, `DATASET_DIR`, `FRACTION`

- [ ] **Step 4: Commit**

```bash
git add scripts/submit_all_experiments_cscs.sh
git commit -m "feat: add submit_all_experiments_cscs.sh for 20-job Slurm batch submission"
```

---

## Task 12: Experiment suite summarizer

**Files:**
- Create: `scripts/summarize_experiment_suite.py`

- [ ] **Step 1: Create the script**

Create `scripts/summarize_experiment_suite.py`:

```python
#!/usr/bin/env python3
"""Summarize all 20 ModernBERT UQ experiment outputs into a single comparison CSV.

Walks outputs/{condition}/frac_{f}/ directories, loads metrics.json files,
and emits a comparison CSV with condition, fraction, and key metrics.

Usage:
    python scripts/summarize_experiment_suite.py \
        --outputs-root outputs \
        --split test
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

CONDITIONS = ["reg_base", "reg_ensemble", "bt_base", "bt_ensemble"]
FRACTIONS = [0.1, 0.25, 0.5, 0.75, 1.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize experiment suite outputs.")
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--split", type=str, default="test", choices=["test", "validation"])
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def frac_to_label(fraction: float) -> str:
    return f"frac_{fraction:.2f}"


def load_metrics(metrics_path: Path) -> dict[str, float]:
    if not metrics_path.exists():
        return {}
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    result: dict[str, float] = {}
    for section, values in payload.items():
        for k, v in values.items():
            result[f"{section}_{k}"] = float(v) if v is not None and not math.isnan(float(v)) else float("nan")
    return result


def load_pairwise_acc(predictions_path: Path) -> float | None:
    """Compute pairwise accuracy from a predictions CSV if score_target column exists."""
    if not predictions_path.exists():
        return None
    try:
        frame = pd.read_csv(predictions_path)
        if "score_target" not in frame.columns or "score_prediction" not in frame.columns:
            return None
        scores = frame["score_prediction"].values
        targets = frame["score_target"].values
        # Sort by target; consecutive pairs with |target_diff| >= 0.1 form pairs
        order = targets.argsort()
        pairs: list[tuple[int, int]] = []
        for i in range(len(order) - 1):
            if targets[order[i + 1]] - targets[order[i]] >= 0.1:
                pairs.append((order[i + 1], order[i]))  # higher target = preferred
        if not pairs:
            return None
        correct = sum(
            1.0 if scores[p] > scores[r] else 0.5 if scores[p] == scores[r] else 0.0
            for p, r in pairs
        )
        return float(correct / len(pairs))
    except Exception:
        return None


def collect_rows(outputs_root: Path, split: str) -> list[dict[str, object]]:
    rows = []
    for condition in CONDITIONS:
        for fraction in FRACTIONS:
            frac_label = frac_to_label(fraction)
            exp_dir = outputs_root / condition / frac_label
            metrics_path = exp_dir / f"{split}_metrics.json"
            predictions_path = exp_dir / f"{split}_predictions.csv"
            if not metrics_path.exists():
                continue
            metrics = load_metrics(metrics_path)
            pairwise_acc = load_pairwise_acc(predictions_path)
            row: dict[str, object] = {
                "condition": condition,
                "fraction": fraction,
                "exp_dir": str(exp_dir),
            }
            row.update(metrics)
            if pairwise_acc is not None:
                row["pairwise_accuracy"] = pairwise_acc
            rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    rows = collect_rows(args.outputs_root.resolve(), args.split)
    if not rows:
        raise SystemExit(f"No completed outputs found under {args.outputs_root}")

    frame = pd.DataFrame(rows).sort_values(["condition", "fraction"])

    out_path = args.out or (args.outputs_root / f"experiment_suite_{args.split}_summary.csv")
    frame.to_csv(out_path, index=False)
    print(f"Wrote {len(frame)} rows to {out_path}")
    print(frame[["condition", "fraction", "score_regression_pearson", "score_regression_rmse"]].to_string())


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the script is importable**

```bash
python -c "import scripts.summarize_experiment_suite" 2>/dev/null || \
python scripts/summarize_experiment_suite.py --help
```

Expected: prints usage help

- [ ] **Step 3: Commit**

```bash
git add scripts/summarize_experiment_suite.py
git commit -m "feat: add summarize_experiment_suite.py for cross-condition comparison"
```

---

## Task 13: Design spec document

**Files:**
- Create: `docs/superpowers/specs/2026-04-13-clariden-experiment-suite-design.md`

- [ ] **Step 1: Create the spec document**

Create `docs/superpowers/specs/2026-04-13-clariden-experiment-suite-design.md` with this content:

```markdown
# Clariden Experiment Suite — Design Spec

**Date:** 2026-04-13  
**Status:** Approved

## Context

Fine-tune ModernBERT on Clariden (CSCS, 4× GH200 per node) for the RuleFollower scoring task.
The bottleneck is not GPU memory (ModernBERT-base fits on one GH200) but experiment throughput.
By requesting 1 GPU per job, Slurm bin-packs 4 experiments per node simultaneously.

## Experiment Matrix

| Condition | Loss | Heads | UQ Source |
|---|---|---|---|
| `reg_base` | MSE | 1 | MC-dropout |
| `reg_ensemble` | MSE (avg) | 5 | Head variance |
| `bt_base` | Bradley-Terry | 1 | MC-dropout |
| `bt_ensemble` | BT (avg) | 5 | Head variance |

Data fractions: 10%, 25%, 50%, 75%, 100% → 20 total experiments.

## Key Design Decisions

**Unfrozen backbone:** All conditions fine-tune the full ModernBERT encoder (unlike RewardUQ which freezes the backbone). This is the primary differentiator from prior work.

**Uniform evaluation:** All conditions evaluate on single-text val/test splits using regression metrics (RMSE, MAE, Pearson, Spearman). BT models predict a scalar reward that can be compared to ground-truth scores directly.

**BT pair generation:** Pairs come from regression scores — any two training examples with |score_i − score_j| ≥ 0.1 form a (preferred, rejected) pair. This avoids needing annotated human preferences.

**Data fraction reproducibility:** Random subsampling uses a fixed NumPy `default_rng(seed)` so results are deterministic across runs.

**Slurm parallelism:** Single GPU per job (`--gpus=1`, `--account=infra01`). Slurm packs 4 per GH200 node automatically. No multi-GPU synchronization needed.

## Data Flow

```
reg_base/  ──────────────────────────────────────► train.csv (MSE conditions)
                                                    validation.csv, test.csv
bt_base/   ← prepare_bt_pairs.py ◄── reg_base/ ► train_pairs.csv (BT conditions)
                                                    validation.csv, test.csv (copied)
```

## New Config Fields

- `loss.type`: `"mse"` (default) or `"bt"`
- `loss.center_rewards_coeff`: float, regularises rewards toward zero (default `0.0`)
- `data.fraction`: float in (0, 1], subsamples train split (default `1.0`)
- `data.pair_score_threshold`: float, min score diff to form a pair (default `0.1`)

## Slurm Account

Use `--account=infra01` for all jobs.
```

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/specs/2026-04-13-clariden-experiment-suite-design.md
git commit -m "docs: add clariden experiment suite design spec"
```

---

## Self-Review Checklist

- [x] **Task 1** covers: `loss_type`, `center_rewards_coeff`, `pair_score_threshold`, `data_fraction` fields
- [x] **Task 2** covers: `bradley_terry_loss()` with gradient test
- [x] **Task 3** covers: `PairDataset`, `PairBatchCollator`, `load_pair_records`, `tokenize_pair_records`
- [x] **Task 4** covers: regression ensemble guard removal + `member_scores` + loss averaging
- [x] **Task 5** covers: `--fraction` CLI flag + `build_dataloader()` subsampling
- [x] **Task 6** covers: `build_pair_dataloader()` + `run_bt_training()` + dispatch in `run_experiment()`
- [x] **Task 7** covers: `pairwise_accuracy()` in `evaluate.py`
- [x] **Task 8** covers: `prepare_bt_pairs.py` script + real data test
- [x] **Task 9** covers: all 4 base configs with correct paths and fields
- [x] **Task 10** covers: `FRACTION` passthrough in container script
- [x] **Task 11** covers: batch submission with dry-run validation
- [x] **Task 12** covers: results summarizer for all 20 experiments
- [x] **Task 13** covers: design doc

**Type consistency check:**
- `member_scores` field added to `UQModelOutput` in Task 4 → referenced correctly in `run_bt_training()` in Task 6 (`out_a.member_scores`)
- `bradley_terry_loss()` defined in Task 2 → imported in `run_bt_training()` in Task 6
- `load_pair_records`, `PairDataset`, `PairBatchCollator` defined in Task 3 → imported in Task 6
- `data_fraction` field added to `ExperimentConfig` in Task 1 → read in `run_experiment()` in Task 5
- `loss_type`, `center_rewards_coeff` added in Task 1 → read in `run_experiment()` in Task 6
- All config paths in Task 9 (`data/rulefollower_heuristic_ann_gpt41mini_*`) match dataset directories in Task 8
