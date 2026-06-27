#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DATASET_DIR="rulefollower_heuristic_ann_gpt41mini_reg_base"

echo "[SBATCH] distribution smoke (1 epoch)"
sbatch \
  --export=ALL,CONFIG_NAME="rulefollower_heuristic_ann_gpt41mini_dist_base_smoke.yaml",DATASET_DIR="$DATASET_DIR" \
  scripts/train_prepared_rulefollower_distribution_cscs_container.sh
echo

echo "[SBATCH] corn smoke (1 epoch)"
sbatch \
  --export=ALL,CONFIG_NAME="rulefollower_heuristic_ann_gpt41mini_corn_base_smoke.yaml",DATASET_DIR="$DATASET_DIR" \
  scripts/train_prepared_rulefollower_ordinal_cscs_container.sh
echo

echo "[DONE] Submitted distribution + CORN smoke jobs."
