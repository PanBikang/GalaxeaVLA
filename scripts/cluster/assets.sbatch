#!/bin/bash
#SBATCH --job-name=g05-assets
#SBATCH --partition=cpu
#SBATCH --qos=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/public/node03/users/panbk/data/GalaxeaVLA/runs/slurm/assets-%j.log
set -euo pipefail
cd /public/node03/users/panbk/data/GalaxeaVLA
source scripts/cluster/env.sh
export HF_HUB_DOWNLOAD_TIMEOUT=180
export HF_HUB_DISABLE_XET=1
python scripts/cluster/download_assets.py
uv pip install --no-deps -e /public/node03/users/panbk/data/third_party/LIBERO
python -c 'from experiments.libero.libero_eval_utils import ensure_libero_import; print(ensure_libero_import())'
