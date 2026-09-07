#!/bin/bash
#SBATCH --job-name=g05-saved-inference
#SBATCH --partition=h20
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --output=/public/node03/users/panbk/data/GalaxeaVLA/runs/slurm/saved-inference-%j.log
set -euo pipefail
cd /public/node03/users/panbk/data/GalaxeaVLA
export MUJOCO_GL=egl
export G05_USE_NATIVE_EGL=1
source scripts/cluster/env.sh
python scripts/cluster/check_saved_inference.py "${1:?Pass the saved checkpoint path}"
