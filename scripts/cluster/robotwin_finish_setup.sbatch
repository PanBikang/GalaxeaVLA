#!/bin/bash
#SBATCH --job-name=g05-robotwin-assets
#SBATCH --partition=cpu
#SBATCH --qos=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/public/node03/users/panbk/data/GalaxeaVLA/runs/slurm/robotwin-assets-%j.log
set -euo pipefail
cd /public/node03/users/panbk/data/GalaxeaVLA
.venv-robotwin/bin/python scripts/cluster/robotwin_assets.py
.venv-robotwin/bin/python - <<'PY'
from importlib.metadata import version
import torch, sapien, sapien.physx, warp, curobo, g05
for package in ['torch','sapien','warp-lang','nvidia-curobo']:
    print(package,version(package),flush=True)
PY
uv pip check --python .venv-robotwin/bin/python
