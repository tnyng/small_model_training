#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

items=(
  "llm_gptoss__ann_gptoss rulefollower_llm_gptoss_ann_gptoss_ge8"
  "llm_gptoss__ann_gpt41mini rulefollower_llm_gptoss_ann_gpt41mini_ge8"
  "llm_gpt5mini__ann_gpt5mini rulefollower_llm_gpt5mini_ann_gpt5mini_ge8"
  "llm_gpt5mini__ann_gpt41mini rulefollower_llm_gpt5mini_ann_gpt41mini_ge8"
  "heuristic__ann_gptoss rulefollower_heuristic_ann_gptoss_ge8"
  "heuristic__ann_gpt41mini rulefollower_heuristic_ann_gpt41mini_ge8"
  "heuristic__ann_gpt5mini rulefollower_heuristic_ann_gpt5mini_ge8"
)

for item in "${items[@]}"; do
  branch="${item%% *}"
  output_name="${item#* }"
  echo "[RUN] $branch -> $output_name"
  python "$ROOT/scripts/prepare_rulefollower_branch.py"     --branch "$branch"     --output-name "$output_name"     --accuracy-threshold 8.0     --label-rule ge
  echo
 done

echo "[DONE] Prepared all RuleFollower GE8 branch datasets."
