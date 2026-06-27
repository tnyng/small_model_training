#!/usr/bin/env bash
#SBATCH --job-name=modernbert_rf_gptoss
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --account=infra01
#SBATCH --output=modernbert_rf_gptoss_%j.out
#SBATCH --error=modernbert_rf_gptoss_%j.err

set -euo pipefail

RULEFOLLOWER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TRAIN_ROOT="${RULEFOLLOWER_ROOT}/small_model_training-main"

cd "${RULEFOLLOWER_ROOT}"

# Adjust this block to match your CSCS environment if needed.
if [ -f "${HOME}/.bashrc" ]; then
  # shellcheck source=/dev/null
  source "${HOME}/.bashrc"
fi

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV:-modernbert_uq}" || true
fi

export PYTHONPATH="${TRAIN_ROOT}/src:${PYTHONPATH:-}"

python small_model_training-main/scripts/prepare_rulefollower_branch.py \
  --branch llm_gptoss__ann_gptoss \
  --output-name rulefollower_llm_gptoss_ann_gptoss_gt8 \
  --accuracy-threshold 8.0

cd "${TRAIN_ROOT}"

accelerate launch -m modernbert_uq.train \
  --config configs/rulefollower_llm_gptoss_ann_gptoss_gt8.yaml
