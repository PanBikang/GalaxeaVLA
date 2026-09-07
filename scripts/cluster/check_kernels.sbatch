#!/bin/bash
#SBATCH --job-name=g05-kernels
#SBATCH --partition=h20
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=/public/node03/users/panbk/data/GalaxeaVLA/runs/slurm/kernels-%j.log
set -euo pipefail
cd /public/node03/users/panbk/data/GalaxeaVLA
source scripts/cluster/env.sh
python -u - <<'PY'
from types import SimpleNamespace
import torch
from g05.models.g05.qwen35.vision import Qwen3_5VisionAttention
from fla.ops.gated_delta_rule import chunk_gated_delta_rule

print('GPU:', torch.cuda.get_device_name(), flush=True)
attention = Qwen3_5VisionAttention(SimpleNamespace(hidden_size=128,num_heads=2)).cuda().bfloat16()
q, k, v = [torch.randn(128,2,64,device='cuda',dtype=torch.bfloat16,requires_grad=True) for _ in range(3)]
cu = torch.tensor([0,64,128],device='cuda',dtype=torch.int32)
y = attention._attend_spatial(q,k,v,cu)
y.float().square().mean().backward()
assert all(torch.isfinite(x.grad).all() for x in [q,k,v])
print('FA4 vision forward/backward PASS', flush=True)
q, k, v = [torch.randn(1,128,2,128,device='cuda',dtype=torch.bfloat16,requires_grad=True) for _ in range(3)]
g = -torch.rand(1,128,2,device='cuda',dtype=torch.float32)
beta = torch.rand(1,128,2,device='cuda',dtype=torch.bfloat16)
y, _ = chunk_gated_delta_rule(q,k,v,g=g,beta=beta,use_qk_l2norm_in_kernel=True)
y.float().square().mean().backward()
assert all(torch.isfinite(x.grad).all() for x in [q,k,v])
print('FLA forward/backward PASS', flush=True)
PY
