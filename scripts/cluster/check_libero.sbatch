#!/bin/bash
#SBATCH --job-name=g05-libero-check
#SBATCH --partition=h20
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=/public/node03/users/panbk/data/GalaxeaVLA/runs/slurm/libero-%j.log
set -euo pipefail
cd /public/node03/users/panbk/data/GalaxeaVLA
export MUJOCO_GL="${MUJOCO_GL:-egl}"
if [[ "$MUJOCO_GL" == egl ]]; then export G05_USE_NATIVE_EGL=1; fi
source scripts/cluster/env.sh
uv pip install --no-deps -e /public/node03/users/panbk/data/third_party/LIBERO
python scripts/cluster/configure_runtime.py
python scripts/cluster/check_libero.py
python -m pytest -q tests/test_serve_policy_dynamic_batching.py
