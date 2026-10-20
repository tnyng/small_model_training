#!/bin/bash
#SBATCH --job-name=prm-v2
#SBATCH --account=ls_lawecon
#SBATCH --partition=gpupr.24h
#SBATCH --time=03:00:00
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
: "${MODE:?MODE required}"
cd "$W"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$VENV/bin/python" train_prm.py \
  --splits splits_v2 --out "runs/$RUNNAME" --mode "$MODE" \
  --exp "${EXP:-gf}" \
  --model answerdotai/ModernBERT-base --bs 32 --epochs 2 --lr 2e-5 --seed 42

mkdir -p "$PERSIST/runs/$RUNNAME" "$PERSIST/logs"
cp -a "runs/$RUNNAME/." "$PERSIST/runs/$RUNNAME/"
cp -f "$W/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" "$PERSIST/logs/" 2>/dev/null || true
cp -f "$W/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err" "$PERSIST/logs/" 2>/dev/null || true
echo "[persist] results copied to $PERSIST/runs/$RUNNAME"
