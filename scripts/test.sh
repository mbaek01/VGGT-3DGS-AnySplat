#!/usr/bin/env bash
# Reuse pytest from dtb-eval without upgrading the pinned native GS environment.
set -euo pipefail
cd "$(dirname "$0")/.."
eval_packages=$(PYTHONNOUSERSITE=1 conda run --no-capture-output -n dtb-eval python -s -c \
  'import site; print(site.getsitepackages()[0])')
PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 conda run --no-capture-output -n dtb-gsplat python -s -c \
  'import sys; sys.path.append(sys.argv[1]); import pytest; raise SystemExit(pytest.main(["tests", "-q"]))' \
  "$eval_packages"
