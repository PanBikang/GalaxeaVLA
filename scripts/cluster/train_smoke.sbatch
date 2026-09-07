#!/bin/bash
#SBATCH --job-name=g05-train-smoke
#SBATCH --partition=h20
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=01:00:00
#SBATCH --output=/public/node03/users/panbk/data/GalaxeaVLA/runs/slurm/train-%j.log
set -euo pipefail
cd /public/node03/users/panbk/data/GalaxeaVLA
export MUJOCO_GL=egl
export G05_USE_NATIVE_EGL=1
source scripts/cluster/env.sh
export EXP_NAME="smoke_${SLURM_JOB_ID}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
bash scripts/run/finetune.sh 1 libero_local --max_datasets 1 --overfit_batch 1 \
    model.batch_size=1 model.max_steps=3 model.num_workers=2 \
    model.warmup_steps=0 logger.mode=offline logger.type=null \
    "datastatics_path=$G05_OUTPUT_DIR/libero_local/$EXP_NAME/smoke_stats.json" \
    eval_steps=100 checkpointing_steps=100
python scripts/cluster/check_saved_inference.py "$G05_OUTPUT_DIR/libero_local/$EXP_NAME/last.pt"
