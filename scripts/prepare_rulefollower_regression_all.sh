#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

items=(
  "heuristic__ann_gpt41mini rulefollower_heuristic_ann_gpt41mini"
  "llm_gpt5mini__ann_gpt41mini rulefollower_llm_gpt5mini_ann_gpt41mini"
  "heuristic__ann_gpt5mini rulefollower_heuristic_ann_gpt5mini"
  "llm_gpt5mini__ann_gpt5mini rulefollower_llm_gpt5mini_ann_gpt5mini"
)
variants=(reg_base reg_down_exact reg_over_exact reg_down_bins reg_over_bins)

for item in "${items[@]}"; do
  branch="${item%% *}"
  prefix="${item#* }"
  for variant in "${variants[@]}"; do
    output_name="${prefix}_${variant}"
    echo "[RUN] $branch -> $output_name"
    python "$ROOT/scripts/prepare_rulefollower_regression.py"       --branch "$branch"       --variant "$variant"       --output-name "$output_name"
    echo
  done
 done

echo "[DONE] Prepared all RuleFollower regression datasets."
