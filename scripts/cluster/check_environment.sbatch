#!/bin/bash
#SBATCH --job-name=g05-env-check
#SBATCH --partition=h20
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=/public/node03/users/panbk/data/GalaxeaVLA/runs/slurm/env-%j.log
set -euo pipefail
cd /public/node03/users/panbk/data/GalaxeaVLA
source scripts/cluster/env.sh
python - <<'PY'
import torch, g05
import flash_attn.cute
import fla
print('torch:', torch.__version__, 'CUDA:', torch.cuda.is_available(), flush=True)
assert '+cu128' in torch.__version__
assert torch.cuda.is_available()
print('GPU:', torch.cuda.get_device_name(), flush=True)
x = torch.randn(256, 256, device='cuda', requires_grad=True)
x.square().mean().backward()
assert torch.isfinite(x.grad).all()
print('CUDA forward/backward PASS', flush=True)
PY
