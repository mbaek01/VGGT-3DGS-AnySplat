#!/usr/bin/env bash
# Linux x86_64, NVIDIA CUDA 12.1 toolkit (nvcc), a compatible driver, conda.
set -euo pipefail
# Keep packages in ~/.local from satisfying dependencies in native environments.
export PYTHONNOUSERSITE=1
export PIP_USER=false
cd "$(dirname "$0")/.."
target="${1:-all}"
case "$target" in
  all|submodules|eval|vggt|gsplat|anysplat) ;;
  -h|--help) echo "Usage: bash scripts/setup.sh [all|submodules|eval|vggt|gsplat|anysplat]"; exit 0 ;;
  *) echo "Unknown target: $target" >&2; exit 2 ;;
esac
if [[ "$target" == all || "$target" == submodules ]]; then
  git submodule update --init --recursive
  if [[ "$target" == submodules ]]; then exit 0; fi
fi
if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
  echo "GPU setup requires Linux x86_64." >&2
  exit 1
fi
if ! command -v conda >/dev/null; then
  echo "Setup failed: conda is not on PATH. Install conda or initialize your shell first." >&2
  exit 1
fi
if [[ "$target" != eval ]] && ! command -v nvcc >/dev/null; then
  echo "Setup failed: nvcc is not on PATH. Install the CUDA 12.1 toolkit or add its bin directory to PATH." >&2
  echo "No environments were created. A GPU driver alone does not include nvcc." >&2
  exit 1
fi
create_env() {
  local name="$1"
  if ! conda run -n "dtb-$name" python --version >/dev/null 2>&1; then
    conda env create -f "envs/$name.yaml"
  fi
}
install_base() {
  local name="$1"
  conda run --no-capture-output -n "dtb-$name" python -m pip install 'pip<25' 'setuptools<75' wheel ninja packaging
}
install_package() {
  local name="$1"
  conda run --no-capture-output -n "dtb-$name" python -m pip install --no-deps -e .
  conda run --no-capture-output -n "dtb-$name" python -m pip check
}
for name in eval vggt gsplat anysplat; do
  if [[ "$target" != all && "$target" != "$name" ]]; then continue; fi
  create_env "$name"
  install_base "$name"
  if [[ "$name" == eval ]]; then
    conda run --no-capture-output -n dtb-eval python -m pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cpu
    conda run --no-capture-output -n dtb-eval python -m pip install -c envs/common-constraints.txt 'numpy==1.26.4' -e '.[eval,test]'
  elif [[ "$name" == vggt ]]; then
    conda run --no-capture-output -n dtb-vggt python -m pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
    conda run --no-capture-output -n dtb-vggt python -m pip install -c envs/vggt-constraints.txt -r third_party/vggt/requirements.txt
    # demo_colmap imports the tracker even when BA is disabled.
    conda run --no-capture-output -n dtb-vggt python -m pip install -c envs/vggt-constraints.txt scipy trimesh tqdm opencv-python hydra-core omegaconf PyYAML pycolmap==3.10.0 pyceres==2.3 'git+https://github.com/jytime/LightGlue.git@2f23ca2ea9638cecad7f7220795210fc6b8353c3'
  elif [[ "$name" == gsplat ]]; then
    conda run --no-capture-output -n dtb-gsplat python -m pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
    conda run --no-capture-output -n dtb-gsplat python -m pip install -c envs/gsplat-constraints.txt 'numpy==1.26.4'
    conda run --no-capture-output -n dtb-gsplat python -m pip install --no-build-isolation -e third_party/gsplat
    conda run --no-capture-output -n dtb-gsplat python -m pip install -c envs/gsplat-constraints.txt -r third_party/gsplat/examples/requirements.txt
  else
    conda run --no-capture-output -n dtb-anysplat python -m pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
    conda run --no-capture-output -n dtb-anysplat python -m pip install -c envs/anysplat-constraints.txt 'numpy==1.25.0' 'torch-scatter==2.1.2' -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
    mkdir -p work
    # Replace only the unpinned PyTorch3D URL; preserve AnySplat's native gsplat 1.4 wheel.
    python3 - <<'PY'
from pathlib import Path
p = Path('third_party/anysplat/requirements.txt').read_text()
p = p.replace('git+https://github.com/facebookresearch/pytorch3d.git', 'git+https://github.com/facebookresearch/pytorch3d.git@f34104cf6ebefacd7b7e07955ee7aaa823e616ac')
Path('work/anysplat-requirements.txt').write_text(p)
PY
    conda run --no-capture-output -n dtb-anysplat python -m pip install --no-build-isolation -c envs/anysplat-constraints.txt -r work/anysplat-requirements.txt
  fi
  install_package "$name"
done
echo "Setup finished. Run the preflight imports in README before launching long jobs."
