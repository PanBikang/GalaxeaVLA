#!/bin/bash
#SBATCH --job-name=g05-robotwin-attention
#SBATCH --partition=h20
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/public/node03/users/panbk/data/GalaxeaVLA/runs/slurm/robotwin-attention-%j.log
set -euo pipefail
cd /public/node03/users/panbk/data/GalaxeaVLA
source scripts/cluster/robotwin_env.sh
python -u - <<'PY'
import torch
from types import SimpleNamespace
from g05.models.g05.qwen35.vision import Qwen3_5VisionAttention
torch.manual_seed(7)
q,k,v=[torch.randn(768,16,64,device='cuda',dtype=torch.bfloat16) for _ in range(3)]
cu=torch.tensor([0,256,512,768],device='cuda',dtype=torch.int32)
attention=Qwen3_5VisionAttention(SimpleNamespace(hidden_size=1024,num_heads=16)).cuda().bfloat16()
with torch.no_grad():
    fa4=attention._attend_spatial(q,k,v,cu)
    attention.attention_backend='sdpa'
    sdpa=attention._attend_spatial(q,k,v,cu)
torch.testing.assert_close(fa4,sdpa,atol=0.01,rtol=0.01)
print('FA4/SDPA max abs difference:',(fa4.float()-sdpa.float()).abs().max().item())
print('VISION ATTENTION EQUIVALENCE PASS')
PY
