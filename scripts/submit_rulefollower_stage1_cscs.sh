#!/usr/bin/env bash
# Stage 1 on the heuristic+gpt41mini branch: fill out the 12 remaining cells so
# we have 3 seeds on reg_base plus 3 alternative slices at seed 42.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

REG_TEMPLATE="scripts/train_prepared_rulefollower_branch_cscs_container.sh"
DIST_TEMPLATE="scripts/train_prepared_rulefollower_distribution_cscs_container.sh"
CORN_TEMPLATE="scripts/train_prepared_rulefollower_ordinal_cscs_container.sh"
ACCOUNT="a0126"

submit() {
  local template="$1" cfg="$2" dataset="$3"
  echo "[SBATCH] ${cfg%.yaml} -> $template"
  sbatch \
    --account=$ACCOUNT \
    --export=ALL,CONFIG_NAME="$cfg",DATASET_DIR="$dataset" \
    "$template"
  echo
}

# --- seed variants on reg_base (3 variants x 2 new seeds = 6) ---
DATASET_BASE="rulefollower_heuristic_ann_gpt41mini_reg_base"

submit "$REG_TEMPLATE"  "rulefollower_heuristic_ann_gpt41mini_reg_base_seed7.yaml"    "$DATASET_BASE"
submit "$REG_TEMPLATE"  "rulefollower_heuristic_ann_gpt41mini_reg_base_seed123.yaml"  "$DATASET_BASE"
submit "$DIST_TEMPLATE" "rulefollower_heuristic_ann_gpt41mini_dist_base_seed7.yaml"   "$DATASET_BASE"
submit "$DIST_TEMPLATE" "rulefollower_heuristic_ann_gpt41mini_dist_base_seed123.yaml" "$DATASET_BASE"
submit "$CORN_TEMPLATE" "rulefollower_heuristic_ann_gpt41mini_corn_base_seed7.yaml"   "$DATASET_BASE"
submit "$CORN_TEMPLATE" "rulefollower_heuristic_ann_gpt41mini_corn_base_seed123.yaml" "$DATASET_BASE"

# --- alternative slices at seed 42 (2 variants x 3 slices = 6) ---
for slice in over_exact down_exact down_bins; do
  DATASET="rulefollower_heuristic_ann_gpt41mini_reg_${slice}"
  submit "$DIST_TEMPLATE" "rulefollower_heuristic_ann_gpt41mini_dist_${slice}.yaml" "$DATASET"
  submit "$CORN_TEMPLATE" "rulefollower_heuristic_ann_gpt41mini_corn_${slice}.yaml" "$DATASET"
done

echo "[DONE] Submitted Stage 1 fan-out (12 jobs) on account ${ACCOUNT}."
