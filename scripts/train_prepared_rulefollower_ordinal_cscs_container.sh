#!/bin/bash -l

#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --account=a0126
#SBATCH --job-name=modernbert_rf_corn
#SBATCH --output=modernbert_rf_corn_%j.out
#SBATCH --error=modernbert_rf_corn_%j.err

set -euo pipefail

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

CONFIG_NAME="${CONFIG_NAME:?Set CONFIG_NAME, e.g. rulefollower_heuristic_ann_gpt41mini_corn_base.yaml}"
DATASET_DIR="${DATASET_DIR:?Set DATASET_DIR, e.g. rulefollower_heuristic_ann_gpt41mini_reg_base}"
JOB_LABEL="${JOB_LABEL:-$DATASET_DIR}"

VENV_DIR="${VENV_DIR:-$SCRATCH/modernbert_uq_venv}"
HF_HOME="${HF_HOME:-$SCRATCH/hf_home}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export TRAIN_ROOT
export VENV_DIR
export HF_HOME
export TRANSFORMERS_CACHE
export PYTHON_BIN

TRAIN_CMD="
cd $TRAIN_ROOT
export PYTHONPATH=\"$TRAIN_ROOT/src:\${PYTHONPATH:-}\"
\"\$VENV_DIR/bin/python\" -m accelerate.commands.launch -m modernbert_uq.ordinal.train \
  --config configs/$CONFIG_NAME
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

if \"\$PYTHON\" - <<'PY'
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
  \"\$PIP\" install pandas pyyaml scikit-learn transformers accelerate
fi

cd \"\$TRAIN_ROOT\"
echo '[INFO] Skipping editable install; using PYTHONPATH for modernbert_uq source.'
"

if [ ! -f "$TRAIN_ROOT/data/$DATASET_DIR/train.csv" ]; then
  echo "[ERROR] Prepared dataset missing: $TRAIN_ROOT/data/$DATASET_DIR/train.csv"
  exit 1
fi

if [ ! -f "$TRAIN_ROOT/configs/$CONFIG_NAME" ]; then
  echo "[ERROR] Config missing: $TRAIN_ROOT/configs/$CONFIG_NAME"
  exit 1
fi

echo "[INFO] TRAIN_ROOT=$TRAIN_ROOT"
echo "[INFO] CONFIG_NAME=$CONFIG_NAME"
echo "[INFO] DATASET_DIR=$DATASET_DIR"
echo "[INFO] JOB_LABEL=$JOB_LABEL"
echo "[INFO] ENTRYPOINT=modernbert_uq.ordinal.train"

if [ "$DEBUG" -eq 0 ]; then
  LAUNCHING="srun --container-writable --environment=$CSCS_ENVIRONMENT"
else
  LAUNCHING=""
fi

$LAUNCHING bash -lc "
  $ENV_BOOTSTRAP
  $TRAIN_CMD
"
