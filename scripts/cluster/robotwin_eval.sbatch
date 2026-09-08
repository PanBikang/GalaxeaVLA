#!/bin/bash
#SBATCH --job-name=g05-robotwin-full
#SBATCH --partition=h20
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=1-00:00:00
#SBATCH --output=/public/node03/users/panbk/data/GalaxeaVLA/runs/slurm/robotwin-full-%j.log
set -euo pipefail
cd /public/node03/users/panbk/data/GalaxeaVLA
source scripts/cluster/robotwin_env.sh
manifest="${1:?Pass experiment manifest}"
base="${2:?Pass first worker index}"
srun --label --gpu-bind=single:1 python -u scripts/cluster/robotwin_worker.py "$manifest" --base-worker "$base"
