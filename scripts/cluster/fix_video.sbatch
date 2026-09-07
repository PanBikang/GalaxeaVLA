#!/bin/bash
#SBATCH --job-name=g05-video-deps
#SBATCH --partition=cpu
#SBATCH --qos=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/public/node03/users/panbk/data/GalaxeaVLA/runs/slurm/video-deps-%j.log
set -euo pipefail
cd /public/node03/users/panbk/data/GalaxeaVLA
source scripts/cluster/env.sh
uv sync --extra dev --index-strategy unsafe-best-match
python scripts/cluster/configure_runtime.py
uv pip install --no-deps -e "$LIBERO_SOURCE_ROOT"
uv pip check
python -c 'from torchcodec.decoders import VideoDecoder; import torchcodec; print("TorchCodec",torchcodec.__version__)'
