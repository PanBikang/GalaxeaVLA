#!/bin/bash
#SBATCH --job-name=g05-libero-train
#SBATCH --partition=h20
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=1-00:00:00
#SBATCH --output=/public/node03/users/panbk/data/GalaxeaVLA/runs/slurm/train-full-%j.log
set -euo pipefail
cd /public/node03/users/panbk/data/GalaxeaVLA
source scripts/cluster/env.sh
export EXP_NAME="${EXP_NAME:-train_${SLURM_JOB_ID}}"
export TRANSFORMERS_OFFLINE=1
bash scripts/run/finetune.sh "${SLURM_GPUS_ON_NODE:-1}" libero_local \
    model.max_steps="${G05_MAX_STEPS:-100000}" \
    model.batch_size="${G05_BATCH_SIZE:-8}" model.num_workers=4 \
    logger.mode=offline "$@"
