#!/bin/bash
#SBATCH --job-name=g05-robotwin-smoke
#SBATCH --partition=h20
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:45:00
#SBATCH --output=/public/node03/users/panbk/data/GalaxeaVLA/runs/slurm/robotwin-smoke-%j.log
set -euo pipefail
cd /public/node03/users/panbk/data/GalaxeaVLA
source scripts/cluster/robotwin_env.sh
uv pip check --python "$VIRTUAL_ENV/bin/python"
python -u experiments/robotwin/eval_robotwin_single.py \
    task=robotwin \
    ckpt=checkpoints/g05-robotwin20/checkpoints/model_state_dict.pt \
    EVALUATION.robotwin_root="$ROBOTWIN_ROOT" \
    EVALUATION.task_name="${2:-click_alarmclock}" \
    EVALUATION.task_config="${1:-demo_clean}" \
    EVALUATION.eval_num_episodes="${3:-2}" \
    EVALUATION.output_dir="runs/robotwin/smoke_${SLURM_JOB_ID}" \
    EVALUATION.replan_steps=24 EVALUATION.timing_enabled=true \
    EVALUATION.vision_attention_backend=sdpa \
    seed=7
