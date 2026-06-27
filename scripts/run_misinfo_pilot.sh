#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$ROOT/.."
python small_model_training-main/scripts/prepare_misinfo_pilot.py

cd "$ROOT"
accelerate launch -m modernbert_uq.train --config configs/misinfo_llm_gpt41mini.yaml
accelerate launch -m modernbert_uq.train --config configs/misinfo_heuristic_gpt41mini.yaml
