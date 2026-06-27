# ModernBERT UQ — Clariden Experiment Suite

Fine-tuning ModernBERT on the Clariden HPC cluster (CSCS, 4× GH200 per node) for the RuleFollower scoring task, comparing 4 training conditions × 5 data fractions = 20 experiments.

## Experiment Matrix

| Condition | Loss | Heads | UQ Source |
|---|---|---|---|
| `reg_base` | MSE | 1 | MC-dropout |
| `reg_ensemble` | MSE (avg) | 5 | Head variance |
| `bt_base` | Bradley-Terry | 1 | MC-dropout |
| `bt_ensemble` | BT (avg) | 5 | Head variance |

Data fractions: **10%, 25%, 50%, 75%, 100%** of the training split (seed-fixed for reproducibility).

---

## Quick Start

### 1. Generate BT pair dataset (once)

```bash
python scripts/prepare_bt_pairs.py \
  --input-dir "${DATA_ROOT:-data}/rulefollower_heuristic_ann_gpt41mini_reg_base" \
  --output-dir "${DATA_ROOT:-data}/rulefollower_heuristic_ann_gpt41mini_bt_base" \
  --threshold 0.1 \
  --max-pairs 2000000
```

### 2. Smoke test a single experiment locally

```bash
PYTHONPATH=src python -m modernbert_uq.train \
  --config configs/reg_base.yaml \
  --data-root "${DATA_ROOT:-data}" \
  --fraction 0.01
```

### 3. Submit all 20 experiments to Clariden

```bash
# If the prepared data lives outside this repo, set DATA_ROOT first.
DATA_ROOT="${DATA_ROOT:-$PWD/../modernbert_uq/data}" \
bash scripts/submit_all_experiments_cscs.sh
```

### 4. Summarise results after jobs complete

```bash
python scripts/summarize_experiment_suite.py \
  --outputs-root outputs \
  --split test
```

---

## What's Implemented

### Training Infrastructure

**`src/modernbert_uq/train.py`** — Accelerate-based training with two modes selected by `config.loss.type`:

- **Regression mode** (`loss.type: mse`): single-text MSE loss + difficulty regression head
- **Bradley-Terry mode** (`loss.type: bt`): pair-based BT loss — two forward passes per step, `−log sigmoid(score_preferred − score_rejected)`, with optional reward-centering regularisation

**`src/modernbert_uq/modeling.py`** — `ModernBertUQModel`:
- Shared ModernBERT encoder (unfrozen, unlike RewardUQ which freezes the backbone)
- Dual heads: score prediction + difficulty regression
- **Head-ensemble UQ**: N independent score heads for regression; variance across members = uncertainty
- `UQModelOutput` fields: `member_scores` (regression ensemble), `member_logits` (classification), `predictive_entropy`, `mutual_information`

**`src/modernbert_uq/losses.py`** — `bradley_terry_loss(score_a, score_b, center_rewards_coeff)`:
```python
loss = -F.logsigmoid(score_a - score_b).mean()
if center_rewards_coeff > 0.0:
    loss = loss + center_rewards_coeff * ((score_a + score_b) ** 2).mean()
```

### Data Pipeline

- **`src/modernbert_uq/data.py`** — `TextDifficultyDataset` + `UQBatchCollator` for regression; `PairDataset` + `PairBatchCollator` for BT pairs
- **`scripts/prepare_bt_pairs.py`** — generates BT pair CSVs from regression scores. Input: `train.csv` with `score_target` column. Output: `train_pairs.csv` with `text_a, text_b, score_a, score_b`. Filters pairs where `|score_a − score_b| ≥ threshold`. With `--max-pairs > 0`, sampling is reproducible and avoids materializing the full `O(n^2)` candidate set.
- **`data/rulefollower_heuristic_ann_gpt41mini_reg_base/`** — regression dataset (~15K train rows, ~3K val/test each)
- **`data/rulefollower_heuristic_ann_gpt41mini_bt_base/`** — generated pair dataset (~2M pairs from threshold=0.1)

### Configuration

Four experiment base configs, each inheriting from `default.yaml`:

| File | Condition | Loss | UQ | Heads |
|---|---|---|---|---|
| `configs/reg_base.yaml` | 1 | MSE | MC-dropout | 1 |
| `configs/reg_ensemble.yaml` | 2 | MSE | head variance | 5 |
| `configs/bt_base.yaml` | 3 | Bradley-Terry | MC-dropout | 1 |
| `configs/bt_ensemble.yaml` | 4 | Bradley-Terry | head variance | 5 |

New config fields in `configs/default.yaml`:

```yaml
loss:
  type: mse              # or: bt
  center_rewards_coeff: 0.0  # BT reward centering (0 = off)
data:
  fraction: 1.0          # subsample training split (0 < f ≤ 1)
  pair_score_threshold: 0.1  # min |score_a - score_b| for pair inclusion
uncertainty:
  method: none            # or: head_ensemble
  num_ensemble_members: 5
  mc_dropout_passes: 16  # used when method: none
```

### Slurm Submission

- **`scripts/run_clariden_debug_rulefollower_llm_gpt5mini_ann_gpt5mini_reg_base.sh`** — direct `srun` smoke run on Clariden `debug`; uses the working single-task launch shape (`--ntasks=1 --mpi=none --gpus=1`) and writes logs under `$SCRATCH`.
- **`scripts/submit_clariden_rulefollower_llm_gpt5mini_ann_gpt5mini_reg_base.sh`** — normal non-debug `sbatch` submission for the same RuleFollower regression task; submits a 1-GPU batch job and launches training via `python -m modernbert_uq.train`.
- **`scripts/train_prepared_rulefollower_branch_cscs_container.sh`** — older generic container-based Slurm job that accepts `CONFIG_NAME` and `DATASET_DIR`; keep this for generic prepared-dataset launches, but prefer the two Clariden-specific scripts above for the `rulefollower_llm_gpt5mini_ann_gpt5mini_reg_base` workflow.
- **`scripts/submit_all_experiments_cscs.sh`** — submits all 20 experiments (4 conditions × 5 fractions). Slurm naturally packs 4 single-GPU jobs per GH200 node.

#### Clariden Usage

On Clariden, run from the project copy under scratch so the container can see the workspace:

```bash
cd /iopsstor/scratch/cscs/jni/modernbert_uq_with_reg_base_data/modernbert_uq
export SCRATCH=/iopsstor/scratch/cscs/jni
```

Debug smoke run:

```bash
./scripts/run_clariden_debug_rulefollower_llm_gpt5mini_ann_gpt5mini_reg_base.sh
```

Normal batch submission:

```bash
./scripts/submit_clariden_rulefollower_llm_gpt5mini_ann_gpt5mini_reg_base.sh
```

Useful overrides:

```bash
ACCOUNT=infra01 \
PARTITION=normal \
TIME_LIMIT=08:00:00 \
JOB_NAME=modernbert_full_run \
./scripts/submit_clariden_rulefollower_llm_gpt5mini_ann_gpt5mini_reg_base.sh
```

Debug partition note: if a new `debug` run stays in `PENDING` with `QOSMaxJobsPerUserLimit`, check `squeue --me` and cancel stale debug jobs before retrying.

### Evaluation

- **`src/modernbert_uq/evaluate.py`** — RMSE, MAE, Pearson, Spearman, `pairwise_accuracy(pairs)` (% of held-out pairs ranked correctly)
- **`scripts/summarize_experiment_suite.py`** — walks `outputs/{condition}/frac_{f}/` tree, emits a flat comparison CSV

---

## Key Design Decisions

**Unfrozen backbone:** Unlike RewardUQ (which freezes the encoder), all conditions fine-tune the full ModernBERT encoder end-to-end.

**Uniform evaluation:** All conditions — including BT-trained models — are evaluated on single-text val/test splits using regression metrics (RMSE, Pearson, etc.). The BT model predicts a scalar reward directly comparable to ground-truth scores.

**Pair generation:** Pairs come from score differences in the regression dataset — no human preference annotations required.

**Slurm parallelism:** Each job requests 1 GPU. Slurm bin-packs 4 jobs per GH200 node automatically — no multi-GPU synchronisation needed.

---

## Repository Layout

```
modernbert_uq/
├── src/modernbert_uq/
│   ├── config.py          # ExperimentConfig + load_config (loss.type, fraction)
│   ├── losses.py          # compute_joint_loss + bradley_terry_loss
│   ├── data.py            # datasets, collators (regression + pair)
│   ├── modeling.py        # ModernBertUQModel (regression ensemble + MC-dropout)
│   ├── train.py           # run_training, run_bt_training, run_experiment
│   ├── evaluate.py         # metrics + pairwise_accuracy
│   ├── uncertainty.py      # MC-dropout
│   └── calibration.py      # temperature scaling
├── configs/
│   ├── default.yaml
│   ├── reg_base.yaml       # Condition 1
│   ├── reg_ensemble.yaml   # Condition 2
│   ├── bt_base.yaml        # Condition 3
│   └── bt_ensemble.yaml    # Condition 4
├── scripts/
│   ├── train_prepared_rulefollower_branch_cscs_container.sh
│   ├── submit_all_experiments_cscs.sh   # batch submit all 20
│   ├── prepare_bt_pairs.py                # generate pair CSV
│   └── summarize_experiment_suite.py       # aggregate results
├── data/
│   ├── rulefollower_heuristic_ann_gpt41mini_reg_base/   # regression
│   └── rulefollower_heuristic_ann_gpt41mini_bt_base/    # BT pairs
└── tests/
    ├── test_bt_loss.py
    ├── test_bt_data.py
    └── test_regression_ensemble.py
```

---

## Tests

Run the pure-Python suite (no GPU required):

```bash
PYTHONPATH=src pytest tests/test_config.py tests/test_bt_loss.py tests/test_bt_data.py tests/test_evaluate.py -v
```

Run the torch suite (requires GPU environment):

```bash
PYTHONPATH=src pytest tests/test_modeling.py tests/test_regression_ensemble.py tests/test_train_smoke.py -v
```

---

## Below: Original ModernBERT UQ Documentation

The following sections describe the core library API and remain relevant.

### Core Library API

```python
from modernbert_uq.config import load_config
from modernbert_uq.modeling import ModernBertUQModel
from modernbert_uq.losses import bradley_terry_loss
from modernbert_uq.evaluate import pairwise_accuracy

config = load_config("configs/bt_ensemble.yaml")

model = ModernBertUQModel(
    model_name=config.model_name,
    dropout_prob=config.dropout_prob,
    score_task_type="regression",   # always "regression" even for BT conditions
    uq_method=config.uq_method,
    num_ensemble_members=config.num_ensemble_members,
)
```

### MC-dropout Inference

```python
from modernbert_uq.uncertainty import mc_dropout_predict

summary = mc_dropout_predict(model, batch, num_passes=8)
# Returns: mean_score_prediction, score_variance, predictive_entropy,
#          mutual_information, mean_difficulty, difficulty_variance
```

### Post-hoc Calibration

```python
from modernbert_uq.calibration import TemperatureScaler, FeatureLogisticCalibrator

temp = TemperatureScaler().fit(logits, labels)
cal = FeatureLogisticCalibrator(use_difficulty=True).fit(
    logits=logits, labels=labels, predicted_difficulty=predicted_difficulty
)
```

### Data Contract

Regression CSV (single-text):

```csv
id,text,score_target,difficulty,split
ex-1,"The sky is blue.",0.875,0.10,train
ex-2,"Penguins can fly.",0.125,0.80,train
```

Pair CSV (`prepare_bt_pairs.py` output):

```csv
text_a,text_b,score_a,score_b
"hello world","goodbye world",0.8,0.2
```

---

## Notes

- `scripts/train.sh`, `scripts/evaluate.sh`, `scripts/predict_mc_dropout.sh` are placeholders — use `python -m modernbert_uq.train --config ...` instead
- `pair_score_threshold` in configs controls pair generation (set at data prep time via `--threshold`)
- `center_rewards_coeff` only applies when `loss.type: bt` — set to `0.0` to disable reward centering
- All 20 Slurm jobs run independently; no checkpoint resume across runs
