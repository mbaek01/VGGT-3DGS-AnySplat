#!/usr/bin/env bash
set -euo pipefail
export PYTHONNOUSERSITE=1
cd "$(dirname "$0")/.."
exec conda run --no-capture-output -n dtb-eval python -m benchmark.run run --scene all --method all "$@"
