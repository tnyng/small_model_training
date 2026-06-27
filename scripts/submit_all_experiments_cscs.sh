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
DATASET_MAP[bt_base]="rulefollower_heuristic_ann_gpt41mini_bt_base_200k"
DATASET_MAP[bt_ensemble]="rulefollower_heuristic_ann_gpt41mini_bt_base_200k"

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
