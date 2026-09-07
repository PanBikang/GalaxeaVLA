#!/bin/bash
#SBATCH --job-name=g05-libero-summary
#SBATCH --partition=cpu
#SBATCH --qos=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=/public/node03/users/panbk/data/GalaxeaVLA/runs/slurm/summary-%j.log
set -euo pipefail
cd /public/node03/users/panbk/data/GalaxeaVLA
source scripts/cluster/env.sh
python scripts/cluster/summarize_libero.py "${1:?Pass the complete evaluation run directory}"
