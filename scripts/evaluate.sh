#!/usr/bin/env bash
set -euo pipefail

echo "Run evaluation after training artifacts are available."
echo "Recommended flow: load validation predictions, compute metrics, then fit calibration."
