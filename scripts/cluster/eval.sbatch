#!/bin/bash
#SBATCH --job-name=g05-libero-eval
#SBATCH --partition=h20
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/public/node03/users/panbk/data/GalaxeaVLA/runs/slurm/eval-%A_%a.log
set -euo pipefail
cd /public/node03/users/panbk/data/GalaxeaVLA
export MUJOCO_GL=egl
export G05_USE_NATIVE_EGL=1
source scripts/cluster/env.sh
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
suites=(libero_spatial libero_object libero_goal libero_10)
suite="${suites[${SLURM_ARRAY_TASK_ID:-0}]}"
trials="${1:-50}"
run_id="${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}"
port=$((20000 + SLURM_JOB_ID % 30000))
bash scripts/run/eval_libero.sh checkpoints/g05-libero/model.pt \
    --suites "$suite" --num_trials "$trials" --num_parallel 10 \
    --max_batch_size 10 --max_wait_ms 50 --port "$port" \
    --seed 7 \
    --output_dir "runs/libero_reproduction/$run_id/$suite"
