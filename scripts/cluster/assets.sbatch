#!/bin/bash
#SBATCH --job-name=g05-assets
#SBATCH --partition=cpu
#SBATCH --qos=cpu
#SBATCH --nodes=4
#SBATCH --ntasks=5
#SBATCH --nodelist=zp-nc[06,12,14,17]
#SBATCH --distribution=cyclic
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/public/node03/users/panbk/data/GalaxeaVLA/runs/slurm/assets-%j.log
set -euo pipefail
cd /public/node03/users/panbk/data/GalaxeaVLA
source scripts/cluster/env.sh
export HF_HUB_DOWNLOAD_TIMEOUT=180
export HF_HUB_DISABLE_XET=1
export G05_DISTRIBUTED_DOWNLOAD=1
export G05_REVISION="${G05_REVISION:-e312be81e90c56a55bcb26b57429bd39a335b449}"
srun --label python -u scripts/cluster/download_assets.py
python - <<'PY'
import json
from pathlib import Path
out = Path('runs/provenance')
entries = {}
for group in range(5):
    for entry in json.loads((out / f'checkpoint_inventory_{group}.json').read_text()):
        entries[entry['path']] = entry
(out / 'checkpoint_inventory.json').write_text(json.dumps(list(entries.values()), indent=2))
print('Complete inventory:', len(entries), 'files,', sum(e['bytes'] for e in entries.values()), 'bytes')
PY
uv pip install --no-deps -e /public/node03/users/panbk/data/third_party/LIBERO
python -c 'from experiments.libero.libero_eval_utils import ensure_libero_import; print(ensure_libero_import())'
