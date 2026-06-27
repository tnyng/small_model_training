#!/bin/bash -l

set -euo pipefail

ACCOUNT="${ACCOUNT:-infra01}"
PARTITION="${PARTITION:-}"
CSCS_ENVIRONMENT="${CSCS_ENVIRONMENT:-qwen3_next}"
CONFIG_NAME="${CONFIG_NAME:-rulefollower_llm_gpt5mini_ann_gpt5mini_reg_base.yaml}"
DATASET_DIR="${DATASET_DIR:-rulefollower_llm_gpt5mini_ann_gpt5mini_reg_base}"
TIME_LIMIT="${TIME_LIMIT:-04:00:00}"
JOB_NAME="${JOB_NAME:-modernbert_rf_gpt5mini_reg_base}"

if [ -n "${TRAIN_ROOT:-}" ]; then
  TRAIN_ROOT="$TRAIN_ROOT"
else
  TRAIN_ROOT="$(pwd)"
fi

SCRATCH="${SCRATCH:?Set SCRATCH, e.g. /iopsstor/scratch/cscs/jni}"
HF_HOME="${HF_HOME:-$SCRATCH/hf_home}"
VENV_DIR="${VENV_DIR:-$SCRATCH/modernbert_uq_venv}"
LOG_OUT="${LOG_OUT:-$SCRATCH/${JOB_NAME}_%j.out}"
LOG_ERR="${LOG_ERR:-$SCRATCH/${JOB_NAME}_%j.err}"

if [ ! -f "$TRAIN_ROOT/configs/$CONFIG_NAME" ]; then
  echo "[ERROR] Missing config: $TRAIN_ROOT/configs/$CONFIG_NAME"
  exit 1
fi

if [ ! -f "$TRAIN_ROOT/data/$DATASET_DIR/train.csv" ]; then
  echo "[ERROR] Missing prepared dataset: $TRAIN_ROOT/data/$DATASET_DIR/train.csv"
  exit 1
fi

SBATCH_ARGS=(
  --account="$ACCOUNT"
  --job-name="$JOB_NAME"
  --time="$TIME_LIMIT"
  --nodes=1
  --ntasks=1
  --cpus-per-task=16
  --gpus=1
  --mem=64G
  --output="$LOG_OUT"
  --error="$LOG_ERR"
  --export=ALL,TRAIN_ROOT="$TRAIN_ROOT",HF_HOME="$HF_HOME",VENV_DIR="$VENV_DIR",CSCS_ENVIRONMENT="$CSCS_ENVIRONMENT",CONFIG_NAME="$CONFIG_NAME"
)

if [ -n "$PARTITION" ]; then
  SBATCH_ARGS+=(--partition="$PARTITION")
fi

echo "[INFO] TRAIN_ROOT=$TRAIN_ROOT"
echo "[INFO] CONFIG_NAME=$CONFIG_NAME"
echo "[INFO] DATASET_DIR=$DATASET_DIR"
echo "[INFO] ACCOUNT=$ACCOUNT"
echo "[INFO] PARTITION=${PARTITION:-<scheduler-default>}"
echo "[INFO] CSCS_ENVIRONMENT=$CSCS_ENVIRONMENT"
echo "[INFO] LOG_OUT=$LOG_OUT"
echo "[INFO] LOG_ERR=$LOG_ERR"

sbatch "${SBATCH_ARGS[@]}" <<'EOF'
#!/bin/bash -l
set -euo pipefail

export SCRATCH="${SCRATCH:?SCRATCH must be set by the login environment}"

srun --ntasks=1 --mpi=none --gpus=1 --cpus-per-task=16 --mem=64G \
  --container-writable --environment="$CSCS_ENVIRONMENT" \
  bash -lc '
    set -euo pipefail
    mkdir -p "$HF_HOME"

    if [ ! -d "$VENV_DIR" ]; then
      python -m venv --system-site-packages "$VENV_DIR"
    fi

    source "$VENV_DIR/bin/activate"
    PIP="$VENV_DIR/bin/pip"
    PYTHON="$VENV_DIR/bin/python"

    "$PIP" install "numpy<2"

    cd "$TRAIN_ROOT"
    export PYTHONPATH="$TRAIN_ROOT/src:${PYTHONPATH:-}"
    export CUDA_VISIBLE_DEVICES=0
    export OMP_NUM_THREADS=1
    export TOKENIZERS_PARALLELISM=false

    "$PYTHON" -m modernbert_uq.train --config "configs/$CONFIG_NAME"
  '
EOF
