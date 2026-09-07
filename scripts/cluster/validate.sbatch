#!/bin/bash
#SBATCH --job-name=g05-config-check
#SBATCH --partition=cpu
#SBATCH --qos=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --output=/public/node03/users/panbk/data/GalaxeaVLA/runs/slurm/validate-%j.log
set -euo pipefail
cd /public/node03/users/panbk/data/GalaxeaVLA
source scripts/cluster/env.sh
for script in scripts/cluster/*.sbatch scripts/cluster/*.sh scripts/run/eval_libero.sh; do bash -n "$script"; done
python - <<'PY'
import ast
from pathlib import Path
from hydra import compose, initialize_config_dir
from g05.utils.config.config_resolvers import register_default_resolvers
register_default_resolvers()
for path in Path('scripts/cluster').glob('*.py'):
    ast.parse(path.read_text())
with initialize_config_dir(config_dir=str(Path('configs').resolve()), version_base='1.3'):
    cfg = compose(config_name='train', overrides=['task=libero_local'])
assert cfg.data.embodiment_datasets.libero.lerobot_ds_version == '2.1'
assert cfg.model.pretrained_ckpt == 'checkpoints/g05-base/checkpoints/model_state_dict.pt'
assert cfg.model.model_arch.discrete_action
for path in cfg.data.embodiment_datasets.libero.dataset_groups[0].dataset_dirs:
    assert (Path(path) / 'meta/info.json').is_file(), path
print('Local Hydra config, data paths, Python and shell syntax PASS')
PY
python -m pytest -q tests/test_serve_policy_dynamic_batching.py
