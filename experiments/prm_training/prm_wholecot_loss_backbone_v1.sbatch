#!/bin/bash
#SBATCH --job-name=prm-backbone-loss
#SBATCH --account=ls_lawecon
#SBATCH --partition=gpupr.24h
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=6G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:40g
#SBATCH --output=/cluster/scratch/yatian/prm_v2/logs/%x_%j.out
#SBATCH --error=/cluster/scratch/yatian/prm_v2/logs/%x_%j.err
set -euo pipefail
W=/cluster/scratch/yatian/prm_v2
PERSIST=/cluster/work/lawecon/Work/yatian/rulefollower-prm/v2
VENV=/cluster/work/lawecon/Work/yufan/legal-brief-dpo/.venv
export HF_HOME=/cluster/work/lawecon/Work/yatian/hf_cache
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
: "${RUNNAME:?RUNNAME required}"
cd "$W"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
if [[ -n "${SMOKE_RUN:-}" ]]; then
  "$VENV/bin/python" verify_gscore_smoke.py "runs/$SMOKE_RUN" "${G_OBJECTIVE:-ce}"
fi
ARGS=(--splits splits_v2 --g-sidecar gscore_sidecar_v1 --out "runs/$RUNNAME"
      --model "${MODEL:-answerdotai/ModernBERT-base}" --max-len 8192 --bs 1
      --grad-accum 8 --epochs "${EPOCHS:-16}" --lr 2e-5 --seed 42
      --f-pos-weight 30 --g-objective "${G_OBJECTIVE:-ce}" --g-loss-weight "${G_LOSS_WEIGHT:-1}")
if [[ "${MAX_TRAIN_COTS:-0}" != 0 ]]; then ARGS+=(--max-train-cots "$MAX_TRAIN_COTS"); fi
if [[ "${MAX_DEV_COTS:-0}" != 0 ]]; then ARGS+=(--max-dev-cots "$MAX_DEV_COTS"); fi
if [[ -n "${EVAL_ONLY:-}" ]]; then
  : "${CHECKPOINT:?CHECKPOINT required with EVAL_ONLY}"
  ARGS+=(--eval-only "$EVAL_ONLY" --checkpoint "$CHECKPOINT")
fi
"$VENV/bin/python" train_wholecot_prm_gscore.py "${ARGS[@]}"
mkdir -p "$PERSIST/runs/$RUNNAME" "$PERSIST/logs"
cp -a "runs/$RUNNAME/." "$PERSIST/runs/$RUNNAME/"
cp -f "$W/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" "$PERSIST/logs/" 2>/dev/null || true
cp -f "$W/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err" "$PERSIST/logs/" 2>/dev/null || true
