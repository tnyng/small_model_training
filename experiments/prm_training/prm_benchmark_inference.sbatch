#!/bin/bash
#SBATCH --job-name=prm-speed-compare
#SBATCH --account=ls_lawecon
#SBATCH --partition=gpupr.24h
#SBATCH --time=02:00:00
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
VENV=/cluster/work/lawecon/Work/yufan/legal-brief-dpo/.venv
PERSIST=/cluster/work/lawecon/Work/yatian/rulefollower-prm/v2
export HF_HOME=/cluster/work/lawecon/Work/yatian/hf_cache
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
cd "$W"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
"$VENV/bin/python" benchmark_prm_inference.py \
  --splits splits_v2 \
  --e2a-checkpoint runs/e2a_sentence_source/checkpoint_epoch1.pt \
  --whole-checkpoint runs/wholecot_binary_r1_unweighted_v1/checkpoint_epoch1.pt \
  --output runs/wholecot_binary_r1_unweighted_v1/speed_vs_e2a.json
cp -f runs/wholecot_binary_r1_unweighted_v1/speed_vs_e2a.json \
  "$PERSIST/runs/wholecot_binary_r1_unweighted_v1/"
