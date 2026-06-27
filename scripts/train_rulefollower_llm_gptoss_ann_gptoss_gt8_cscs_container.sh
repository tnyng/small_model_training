#!/bin/bash -l

#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --account=infra01
#SBATCH --job-name=modernbert_rf_gptoss
#SBATCH --output=modernbert_rf_gptoss_%j.out
#SBATCH --error=modernbert_rf_gptoss_%j.err

set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
TRAIN_ROOT="$ROOT/small_model_training-main"

DEBUG="${DEBUG:-0}"
CSCS_ENVIRONMENT="${CSCS_ENVIRONMENT:-qwen3_next}"
PYTHON_BIN="${PYTHON_BIN:-python}"

# Reuse a persistent virtualenv under SCRATCH so we do not reinstall on every job.
VENV_DIR="${VENV_DIR:-$SCRATCH/modernbert_uq_venv}"
HF_HOME="${HF_HOME:-$SCRATCH/hf_home}"
export HF_HOME
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"

PREP_CMD="
cd $ROOT
$PYTHON_BIN small_model_training-main/scripts/prepare_rulefollower_branch.py \
  --branch llm_gptoss__ann_gptoss \
  --output-name rulefollower_llm_gptoss_ann_gptoss_gt8 \
  --accuracy-threshold 8.0
"

TRAIN_CMD="
cd $TRAIN_ROOT
accelerate launch -m modernbert_uq.train \
  --config configs/rulefollower_llm_gptoss_ann_gptoss_gt8.yaml
"

ENV_BOOTSTRAP="
set -euo pipefail
mkdir -p \"$HF_HOME\"

if [ ! -d \"$VENV_DIR\" ]; then
  $PYTHON_BIN -m venv \"$VENV_DIR\"
fi

source \"$VENV_DIR/bin/activate\"
python -m pip install --upgrade pip setuptools wheel

# Install runtime deps if missing. Torch is included because this training job needs it.
if python - <<'PY'
import importlib
mods = ['torch', 'transformers', 'accelerate', 'pandas', 'yaml', 'sklearn']
missing = [m for m in mods if importlib.util.find_spec(m) is None]
print('MISSING:', missing)
raise SystemExit(1 if missing else 0)
PY
then
  echo '[INFO] Runtime dependencies already available in the container/venv.'
else
  pip install torch transformers accelerate pandas pyyaml scikit-learn numpy
fi

cd \"$TRAIN_ROOT\"
pip install -e .
"

if [ "$DEBUG" -eq 0 ]; then
  LAUNCHING="srun --container-writable --environment=$CSCS_ENVIRONMENT"
else
  LAUNCHING=""
fi

$LAUNCHING bash -lc "
  $ENV_BOOTSTRAP
  $PREP_CMD
  $TRAIN_CMD
"
