#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

items=(
  "rulefollower_heuristic_ann_gpt41mini_reg_base"
  "rulefollower_heuristic_ann_gpt41mini_reg_down_exact"
  "rulefollower_heuristic_ann_gpt41mini_reg_over_exact"
  "rulefollower_heuristic_ann_gpt41mini_reg_down_bins"
  "rulefollower_heuristic_ann_gpt41mini_reg_over_bins"
  "rulefollower_llm_gpt5mini_ann_gpt41mini_reg_base"
  "rulefollower_llm_gpt5mini_ann_gpt41mini_reg_down_exact"
  "rulefollower_llm_gpt5mini_ann_gpt41mini_reg_over_exact"
  "rulefollower_llm_gpt5mini_ann_gpt41mini_reg_down_bins"
  "rulefollower_llm_gpt5mini_ann_gpt41mini_reg_over_bins"
  "rulefollower_heuristic_ann_gpt5mini_reg_base"
  "rulefollower_heuristic_ann_gpt5mini_reg_down_exact"
  "rulefollower_heuristic_ann_gpt5mini_reg_over_exact"
  "rulefollower_heuristic_ann_gpt5mini_reg_down_bins"
  "rulefollower_heuristic_ann_gpt5mini_reg_over_bins"
  "rulefollower_llm_gpt5mini_ann_gpt5mini_reg_base"
  "rulefollower_llm_gpt5mini_ann_gpt5mini_reg_down_exact"
  "rulefollower_llm_gpt5mini_ann_gpt5mini_reg_over_exact"
  "rulefollower_llm_gpt5mini_ann_gpt5mini_reg_down_bins"
  "rulefollower_llm_gpt5mini_ann_gpt5mini_reg_over_bins"
)

for name in "${items[@]}"; do
  echo "[SBATCH] $name"
  sbatch --export=ALL,CONFIG_NAME="${name}.yaml",DATASET_DIR="$name"     scripts/train_prepared_rulefollower_branch_cscs_container.sh
  echo
 done

echo "[DONE] Submitted all RuleFollower regression jobs."
