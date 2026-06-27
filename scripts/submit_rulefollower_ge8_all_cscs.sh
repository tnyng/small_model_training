#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

items=(
  "rulefollower_llm_gptoss_ann_gptoss_ge8.yaml rulefollower_llm_gptoss_ann_gptoss_ge8"
  "rulefollower_llm_gptoss_ann_gpt41mini_ge8.yaml rulefollower_llm_gptoss_ann_gpt41mini_ge8"
  "rulefollower_llm_gpt5mini_ann_gpt5mini_ge8.yaml rulefollower_llm_gpt5mini_ann_gpt5mini_ge8"
  "rulefollower_llm_gpt5mini_ann_gpt41mini_ge8.yaml rulefollower_llm_gpt5mini_ann_gpt41mini_ge8"
  "rulefollower_heuristic_ann_gptoss_ge8.yaml rulefollower_heuristic_ann_gptoss_ge8"
  "rulefollower_heuristic_ann_gpt41mini_ge8.yaml rulefollower_heuristic_ann_gpt41mini_ge8"
  "rulefollower_heuristic_ann_gpt5mini_ge8.yaml rulefollower_heuristic_ann_gpt5mini_ge8"
)

for item in "${items[@]}"; do
  config_name="${item%% *}"
  dataset_dir="${item#* }"
  echo "[SBATCH] $config_name | $dataset_dir"
  sbatch --export=ALL,CONFIG_NAME="$config_name",DATASET_DIR="$dataset_dir"     scripts/train_prepared_rulefollower_branch_cscs_container.sh
  echo
 done

echo "[DONE] Submitted all RuleFollower GE8 jobs."
