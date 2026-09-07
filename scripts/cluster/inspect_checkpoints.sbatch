#!/bin/bash
#SBATCH --job-name=g05-inspect-weights
#SBATCH --partition=cpu
#SBATCH --qos=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/public/node03/users/panbk/data/GalaxeaVLA/runs/slurm/inspect-%j.log
set -euo pipefail
cd /public/node03/users/panbk/data/GalaxeaVLA
source scripts/cluster/env.sh
python scripts/cluster/inspect_checkpoints.py
