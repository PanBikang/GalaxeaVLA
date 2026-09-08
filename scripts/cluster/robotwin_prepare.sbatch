#!/bin/bash
#SBATCH --job-name=g05-robotwin-plan
#SBATCH --partition=cpu
#SBATCH --qos=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/public/node03/users/panbk/data/GalaxeaVLA/runs/slurm/robotwin-plan-%j.log
set -euo pipefail
cd /public/node03/users/panbk/data/GalaxeaVLA
source scripts/cluster/robotwin_env.sh
python scripts/cluster/robotwin_protocol.py --run-id robotwin_20260908 --workers 32
uv pip check --python "$VIRTUAL_ENV/bin/python"
mkdir -p runs/tests
python -m pytest -q --basetemp="runs/tests/robotwin-plan-$SLURM_JOB_ID" tests/test_robotwin_runner.py
for script in scripts/cluster/robotwin*.sbatch scripts/cluster/robotwin_env.sh; do bash -n "$script"; done
