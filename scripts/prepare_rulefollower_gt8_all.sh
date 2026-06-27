#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

declare -a BRANCH_TO_OUTPUT=(
  "llm_gptoss__ann_gptoss rulefollower_llm_gptoss_ann_gptoss_gt8"
  "llm_gptoss__ann_gpt41mini rulefollower_llm_gptoss_ann_gpt41mini_gt8"
  "llm_gpt5mini__ann_gpt5mini rulefollower_llm_gpt5mini_ann_gpt5mini_gt8"
  "llm_gpt5mini__ann_gpt41mini rulefollower_llm_gpt5mini_ann_gpt41mini_gt8"
  "heuristic__ann_gptoss rulefollower_heuristic_ann_gptoss_gt8"
  "heuristic__ann_gpt41mini rulefollower_heuristic_ann_gpt41mini_gt8"
  "heuristic__ann_gpt5mini rulefollower_heuristic_ann_gpt5mini_gt8"
)

for item in "${BRANCH_TO_OUTPUT[@]}"; do
  branch="${item%% *}"
  output_name="${item#* }"
  echo "[RUN] $branch -> $output_name"
  python "$ROOT/scripts/prepare_rulefollower_branch.py" \
    --branch "$branch" \
    --output-name "$output_name" \
    --accuracy-threshold 8.0
done

echo "[DONE] Prepared all RuleFollower branch datasets."
