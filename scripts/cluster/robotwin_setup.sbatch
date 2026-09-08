#!/bin/bash
#SBATCH --job-name=g05-robotwin-setup
#SBATCH --partition=cpu
#SBATCH --qos=cpu
#SBATCH --nodelist=zp-nc06
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/public/node03/users/panbk/data/GalaxeaVLA/runs/slurm/robotwin-setup-%j.log
set -euo pipefail
cd /public/node03/users/panbk/data/GalaxeaVLA
export PATH=/public/node03/users/panbk/.local/bin:$PATH
export CUDA_HOME=/public/node03/users/panbk/miniconda3/envs/fastwam
export PATH="$CUDA_HOME/bin:$PATH"
export CPATH="$CUDA_HOME/targets/x86_64-linux/include:${CPATH:-}"
export LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib:${LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST=9.0
export MAX_JOBS=8
export OMP_NUM_THREADS=1
mkdir -p third_party
if [[ ! -d third_party/RoboTwin/.git ]]; then
    git init third_party/RoboTwin
    git -C third_party/RoboTwin remote add origin https://github.com/RoboTwin-Platform/RoboTwin.git
fi
if [[ "$(git -C third_party/RoboTwin rev-parse HEAD 2>/dev/null)" != bf44be51cf5717a5595ce59447f2cf5263d2aa95 ]]; then
    git -C third_party/RoboTwin fetch --depth=1 origin bf44be51cf5717a5595ce59447f2cf5263d2aa95
    git -C third_party/RoboTwin switch --detach bf44be51cf5717a5595ce59447f2cf5263d2aa95
fi
git -C third_party/RoboTwin rev-parse HEAD
if [[ ! -d third_party/curobo/.git ]]; then
    git clone --depth=1 --branch v0.7.5 https://github.com/NVlabs/curobo.git third_party/curobo
fi
UV_PROJECT_ENVIRONMENT=.venv-robotwin uv sync --no-default-groups --extra dev --extra robotwin --index-strategy unsafe-best-match
uv pip install --python .venv-robotwin/bin/python --index-url https://pypi.org/simple \
    'sapien==3.0.0b1' 'warp-lang==0.11.0' 'pip==25.3' 'vcs-versioning==2.2.0'
.venv-robotwin/bin/python -m pip install --no-build-isolation --no-deps ./third_party/curobo
.venv-robotwin/bin/python scripts/cluster/robotwin_assets.py
.venv-robotwin/bin/python - <<'PY'
from importlib.metadata import version
import torch, sapien, sapien.physx, warp, curobo, g05
for package in ['torch','sapien','warp-lang','nvidia-curobo']:
    print(package,version(package),flush=True)
PY
