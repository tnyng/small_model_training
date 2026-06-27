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

# On Slurm, the submitted script is copied into /var/spool/..., so
# BASH_SOURCE[0] points to the spool copy rather than the uploaded repo.
# Prefer the submit directory and allow an explicit override.
if [ -n "${TRAIN_ROOT:-}" ]; then
  TRAIN_ROOT="$TRAIN_ROOT"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  TRAIN_ROOT="$SLURM_SUBMIT_DIR"
else
  TRAIN_ROOT="$(pwd)"
fi

DEBUG="${DEBUG:-0}"
CSCS_ENVIRONMENT="${CSCS_ENVIRONMENT:-qwen3_next}"
PYTHON_BIN="${PYTHON_BIN:-python}"

VENV_DIR="${VENV_DIR:-$SCRATCH/modernbert_uq_venv}"
HF_HOME="${HF_HOME:-$SCRATCH/hf_home}"
TRAIN_ROOT="$TRAIN_ROOT"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export TRAIN_ROOT
export VENV_DIR
export HF_HOME
export TRANSFORMERS_CACHE
export PYTHON_BIN

TRAIN_CMD="
cd $TRAIN_ROOT
export PYTHONPATH=\"$TRAIN_ROOT/src:\${PYTHONPATH:-}\"
\"\$VENV_DIR/bin/python\" -m accelerate.commands.launch -m modernbert_uq.train \
  --config configs/rulefollower_llm_gptoss_ann_gptoss_gt8.yaml
"

ENV_BOOTSTRAP="
set -euo pipefail
mkdir -p \"\$HF_HOME\"

if [ ! -d \"\$VENV_DIR\" ]; then
  $PYTHON_BIN -m venv --system-site-packages \"\$VENV_DIR\"
fi

source \"\$VENV_DIR/bin/activate\"
PYTHON=\"\$VENV_DIR/bin/python\"
PIP=\"\$VENV_DIR/bin/pip\"

\"\$PYTHON\" -m pip install --upgrade pip setuptools wheel
\"\$PIP\" install 'numpy<2'

if "\$PYTHON" - <<'PY'
mods = ['torch', 'transformers', 'accelerate', 'pandas', 'yaml', 'sklearn']
missing = []
for mod in mods:
    try:
        __import__(mod)
    except Exception:
        missing.append(mod)
print('MISSING:', missing)
raise SystemExit(1 if missing else 0)
PY
then
  echo '[INFO] Runtime dependencies already available in the container/venv.'
else
  "\$PIP" install pandas pyyaml scikit-learn transformers accelerate
fi

cd \"\$TRAIN_ROOT\"
\"\$PIP\" install --no-deps -e .
"

if [ ! -f "$TRAIN_ROOT/data/rulefollower_llm_gptoss_ann_gptoss_gt8/train.csv" ]; then
  echo "[ERROR] Prepared dataset missing: $TRAIN_ROOT/data/rulefollower_llm_gptoss_ann_gptoss_gt8/train.csv"
  echo "[ERROR] Upload small_model_training-main with the prepared data folder included."
  exit 1
fi

if [ "$DEBUG" -eq 0 ]; then
  LAUNCHING="srun --container-writable --environment=$CSCS_ENVIRONMENT"
else
  LAUNCHING=""
fi

$LAUNCHING bash -lc "
  $ENV_BOOTSTRAP
  $TRAIN_CMD
"
