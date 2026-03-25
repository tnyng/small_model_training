#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/default.yaml}"

echo "Launch training with Accelerate using ${CONFIG_PATH}"
echo "Example: accelerate launch -m modernbert_uq.train --config ${CONFIG_PATH}"
