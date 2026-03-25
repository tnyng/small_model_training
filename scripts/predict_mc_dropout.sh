#!/usr/bin/env bash
set -euo pipefail

echo "Run MC-dropout prediction on a trained checkpoint."
echo "Recommended flow: keep dropout active, collect multiple passes, then aggregate uncertainty summaries."
