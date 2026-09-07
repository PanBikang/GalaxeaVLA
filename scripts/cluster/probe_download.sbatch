#!/bin/bash
#SBATCH --job-name=g05-download-probe
#SBATCH --partition=cpu
#SBATCH --qos=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=/public/node03/users/panbk/data/GalaxeaVLA/runs/slurm/download-probe-%j.log
set -euo pipefail
cd /public/node03/users/panbk/data/GalaxeaVLA
source scripts/cluster/env.sh
python -u - <<'PY'
import time
from concurrent.futures import ThreadPoolExecutor
import requests
from huggingface_hub import get_token
def probe(host):
    url=f'https://{host}/OpenGalaxea/G05/resolve/main/g05-robotwin20/checkpoints/model_state_dict.pt'
    start=time.monotonic()
    try:
        with requests.get(url, headers={'Authorization': f'Bearer {get_token()}', 'Range':'bytes=0-8388607'}, stream=True, timeout=60) as r:
            r.raise_for_status()
            size=0
            for block in r.iter_content(1024*1024):
                size+=len(block)
                if size>=8388608: break
            elapsed=time.monotonic()-start
            return host, r.status_code, size, round(elapsed,2), round(size/elapsed/1024**2,2)
    except Exception as e:
        return host, type(e).__name__, str(e).split('?')[0]
for result in ThreadPoolExecutor(2).map(probe,['huggingface.co','hf-mirror.com']): print(result, flush=True)
PY
