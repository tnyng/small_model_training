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

**BT pair generation:** Pairs come from regression scores — any two training examples with |score_i − score_j| ≥ 0.1 form a (preferred, rejected) pair. This avoids needing annotated human preferences. For large datasets, the generator samples reproducibly when `--max-pairs > 0` so setup remains tractable and does not need to materialize the full `O(n^2)` candidate set.

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
