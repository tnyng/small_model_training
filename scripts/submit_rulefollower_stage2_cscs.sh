#!/usr/bin/env bash
# Stage 2 ablations on reg_base, seed 42:
#   - soft-label sigma sweep (dist): sigma in {0.5, 1.0, 1.5, 2.0}
#   - difficulty head on (dist, corn): difficulty_weight = 1.0
#   - MC dropout with 20 passes (dist, corn): check whether 10 was enough
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DIST_TEMPLATE="scripts/train_prepared_rulefollower_distribution_cscs_container.sh"
CORN_TEMPLATE="scripts/train_prepared_rulefollower_ordinal_cscs_container.sh"
ACCOUNT="a0126"
DATASET="rulefollower_heuristic_ann_gpt41mini_reg_base"

submit() {
  local template="$1" cfg="$2"
  echo "[SBATCH] ${cfg%.yaml} -> $template"
  sbatch \
    --account=$ACCOUNT \
    --export=ALL,CONFIG_NAME="$cfg",DATASET_DIR="$DATASET" \
    "$template"
  echo
}

# sigma sweep (4) --- dist only, difficulty off
for sig in sigma05 sigma10 sigma15 sigma20; do
  submit "$DIST_TEMPLATE" "rulefollower_heuristic_ann_gpt41mini_dist_base_${sig}.yaml"
done

# difficulty head on (2)
submit "$DIST_TEMPLATE" "rulefollower_heuristic_ann_gpt41mini_dist_base_diff_on.yaml"
submit "$CORN_TEMPLATE" "rulefollower_heuristic_ann_gpt41mini_corn_base_diff_on.yaml"

# MC dropout 20 passes (2)
submit "$DIST_TEMPLATE" "rulefollower_heuristic_ann_gpt41mini_dist_base_mc20.yaml"
submit "$CORN_TEMPLATE" "rulefollower_heuristic_ann_gpt41mini_corn_base_mc20.yaml"

echo "[DONE] Submitted Stage 2 fan-out (8 jobs) on account ${ACCOUNT}."
