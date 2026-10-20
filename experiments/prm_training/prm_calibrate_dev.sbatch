#!/bin/bash
#SBATCH --job-name=prm-dev-calibrate
#SBATCH --account=ls_lawecon
#SBATCH --partition=normal.4h
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --output=/cluster/scratch/yatian/prm_v2/logs/%x_%j.out
#SBATCH --error=/cluster/scratch/yatian/prm_v2/logs/%x_%j.err
set -euo pipefail

W=/cluster/scratch/yatian/prm_v2
VENV=/cluster/work/lawecon/Work/yufan/legal-brief-dpo/.venv
PERSIST=/cluster/work/lawecon/Work/yatian/rulefollower-prm/v2
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
cd "$W"
for run in wholecot_binary_r1_unweighted_v1 wholecot_binary_r2_fpos10_v1 wholecot_binary_r2_fpos30_v1; do
  "$VENV/bin/python" calibrate_prm_dev.py \
    --predictions "runs/$run/dev_predictions_epoch1.jsonl" \
    --output "runs/$run/calibration_dev.json" \
    --min-triggers 20
  cp -f "runs/$run/calibration_dev.json" "$PERSIST/runs/$run/"
done
