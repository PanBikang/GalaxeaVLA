#!/bin/bash
#SBATCH --job-name=g05-robotwin-tests
#SBATCH --partition=cpu
#SBATCH --qos=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --output=/public/node03/users/panbk/data/GalaxeaVLA/runs/slurm/robotwin-tests-%j.log
set -euo pipefail
cd /public/node03/users/panbk/data/GalaxeaVLA
mkdir -p runs/tests
.venv/bin/python -m pytest -q --basetemp="runs/tests/robotwin-$SLURM_JOB_ID" tests/test_robotwin_runner.py
