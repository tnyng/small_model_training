#!/bin/bash
#SBATCH --job-name=prm-e5
#SBATCH --account=ls_lawecon
#SBATCH --partition=gpupr.24h
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=6G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:20g
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

cd "$W"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$VENV/bin/python" run_e5_quote_corruption.py \
  --test splits_v2/test.jsonl \
  --dev-predictions "$PERSIST/runs/e2a_sentence_source/dev_predictions_epoch1.jsonl" \
  --checkpoint "$PERSIST/runs/e2a_sentence_source/checkpoint_epoch1.pt" \
  --tokenizer "$PERSIST/runs/e2a_sentence_source/tokenizer" \
  --out runs/e5_quote_corruption \
  --limit 300 --seed 42 --batch-size 32

mkdir -p "$PERSIST/runs/e5_quote_corruption" "$PERSIST/logs"
cp -a runs/e5_quote_corruption/. "$PERSIST/runs/e5_quote_corruption/"
cp -f "$W/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" "$PERSIST/logs/" 2>/dev/null || true
cp -f "$W/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err" "$PERSIST/logs/" 2>/dev/null || true
echo "[persist] E5 results copied to $PERSIST/runs/e5_quote_corruption"
