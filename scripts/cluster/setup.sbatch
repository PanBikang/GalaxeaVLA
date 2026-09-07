#!/bin/bash
#SBATCH --job-name=g05-setup
#SBATCH --partition=cpu
#SBATCH --qos=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/public/node03/users/panbk/data/GalaxeaVLA/runs/slurm/setup-%j.log
set -euo pipefail
cd /public/node03/users/panbk/data/GalaxeaVLA
export PATH=/public/node03/users/panbk/.local/bin:$PATH
export UV_CACHE_DIR=/public/node03/users/panbk/.cache/uv
export UV_HTTP_TIMEOUT=180
export UV_CONCURRENT_DOWNLOADS=8
export MAX_JOBS=8
uv sync --frozen --extra dev --index-strategy unsafe-best-match
.venv/bin/python -c 'import torch, g05, transformers; print("torch", torch.__version__, "transformers", transformers.__version__)'
uv pip check
python scripts/cluster/configure_runtime.py
