#!/bin/bash
#SBATCH --job-name=g05-robotwin-render
#SBATCH --partition=h20
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/public/node03/users/panbk/data/GalaxeaVLA/runs/slurm/robotwin-render-%j.log
set -euo pipefail
cd /public/node03/users/panbk/data/GalaxeaVLA
source scripts/cluster/robotwin_env.sh
python -u scripts/cluster/robotwin_render.py
