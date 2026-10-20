#!/bin/bash
#SBATCH --job-name=prm-wholecot-v1
#SBATCH --account=ls_lawecon
#SBATCH --partition=gpupr.24h
#SBATCH --time=06:00:00
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
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

: "${RUNNAME:?RUNNAME required}"
cd "$W"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader

ARGS=(--splits splits_v2 --out "runs/$RUNNAME" --model answerdotai/ModernBERT-base
      --max-len "${MAX_LEN:-8192}" --bs "${BS:-1}" --grad-accum "${GRAD_ACCUM:-8}"
      --epochs "${EPOCHS:-2}" --lr 2e-5 --seed 42
      --f-pos-weight "${F_POS_WEIGHT:-1}")
if [[ "${PROFILE_ONLY:-0}" == 1 ]]; then ARGS+=(--profile-only); fi
if [[ "${NO_CHECKPOINT:-0}" == 1 ]]; then ARGS+=(--no-checkpoint); fi
if [[ -n "${CHECKPOINT_POLICY:-}" ]]; then ARGS+=(--checkpoint-policy "$CHECKPOINT_POLICY"); fi
if [[ "${MAX_TRAIN_COTS:-0}" != 0 ]]; then ARGS+=(--max-train-cots "$MAX_TRAIN_COTS"); fi
if [[ "${MAX_DEV_COTS:-0}" != 0 ]]; then ARGS+=(--max-dev-cots "$MAX_DEV_COTS"); fi

"$VENV/bin/python" "${TRAIN_SCRIPT:-train_wholecot_prm.py}" "${ARGS[@]}"

mkdir -p "$PERSIST/runs/$RUNNAME" "$PERSIST/logs"
cp -a "runs/$RUNNAME/." "$PERSIST/runs/$RUNNAME/"
cp -f "$W/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" "$PERSIST/logs/" 2>/dev/null || true
cp -f "$W/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err" "$PERSIST/logs/" 2>/dev/null || true
