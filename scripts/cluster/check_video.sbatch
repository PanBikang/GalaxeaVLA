#!/bin/bash
#SBATCH --job-name=g05-video-check
#SBATCH --partition=cpu
#SBATCH --qos=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --output=/public/node03/users/panbk/data/GalaxeaVLA/runs/slurm/video-%j.log
set -euo pipefail
cd /public/node03/users/panbk/data/GalaxeaVLA
source scripts/cluster/env.sh
python -u - <<'PY'
import os
from pathlib import Path
import torch
from g05.data.lerobot.datasets.video_utils import decode_video_frames, get_safe_default_codec
print('Video backend:', get_safe_default_codec())
for suite in ['libero_10','libero_goal','libero_object','libero_spatial']:
    root=Path(os.environ['LIBERO_DATA_ROOT']) / f'{suite}_no_noops_lerobot'
    for camera in ['observation.images.image','observation.images.wrist_image']:
        video=next(root.glob(f'videos/chunk-*/{camera}/episode_*.mp4'))
        frames=decode_video_frames(video,[0.0,0.05],tolerance_s=0.02)
        assert tuple(frames.shape)==(2,3,512,512)
        assert torch.isfinite(frames).all()
        print(suite,camera,tuple(frames.shape),'PASS',flush=True)
PY
