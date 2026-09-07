"""Fetch all official G05 bundles, pinned to one HF revision."""
import json
import os
from pathlib import Path

import requests
from huggingface_hub import HfApi, hf_hub_download
from ranged_download import download_large

root = Path(__file__).resolve().parents[2]
out = root / 'runs' / 'provenance'
out.mkdir(parents=True, exist_ok=True)
api = HfApi()
revision = os.environ.get('G05_REVISION') or api.model_info('OpenGalaxea/G05').sha
group = int(os.environ['SLURM_PROCID']) if os.environ.get('G05_DISTRIBUTED_DOWNLOAD') else None
bundles = ['g05-libero', 'g05-base', 'g05-robotwin20', 'g05-droid', 'g05-so101']
if group in (None, 0):
    (out / 'hf_revision.json').write_text(json.dumps({'repo': 'OpenGalaxea/G05', 'revision': revision}, indent=2))
print('Downloading OpenGalaxea/G05 revision', revision, flush=True)
entries = [e for e in api.list_repo_tree('OpenGalaxea/G05', revision=revision, recursive=True)
           if hasattr(e, 'size')]
if group is not None:
    entries = [e for e in entries if e.size < 1024**3 or e.path.startswith(bundles[group] + '/')]
for entry in entries:
    if entry.size < 1024**3:
        local = root / 'checkpoints' / entry.path
        if not local.is_file() or local.stat().st_size != entry.size:
            hf_hub_download('OpenGalaxea/G05', entry.path, revision=revision, local_dir=root / 'checkpoints')
priority = {'g05-libero': 0, 'g05-base': 1, 'g05-robotwin20': 2, 'g05-droid': 3, 'g05-so101': 4}
for entry in sorted((e for e in entries if e.size >= 1024**3),
                    key=lambda e: priority.get(e.path.split('/')[0], 99)):
    download_large('OpenGalaxea/G05', revision, entry, root / 'checkpoints' / entry.path, workers=16)
inventory = []
for entry in entries:
    if hasattr(entry, 'size'):
        local = root / 'checkpoints' / entry.path
        assert local.is_file(), entry.path
        assert local.stat().st_size == entry.size, entry.path
        inventory.append({'path': entry.path, 'bytes': entry.size})
inventory_name = 'checkpoint_inventory.json' if group is None else f'checkpoint_inventory_{group}.json'
(out / inventory_name).write_text(json.dumps(inventory, indent=2))
print('Verified', len(inventory), 'files;', sum(x['bytes'] for x in inventory), 'bytes', flush=True)
if group in (None, 0):
    for name, target in [('action_tokenizer.pt', '../action_tokenizer.pt'),
                         ('hf_processor', '../qwen3_5_2b_base_processor')]:
        link = root / 'checkpoints' / 'g05-libero' / name
        if not link.exists() and not link.is_symlink():
            link.symlink_to(target)
    response = requests.get('https://arxiv.org/html/2608.11739v1', timeout=60)
    response.raise_for_status()
    (out / 'paper.html').write_text(response.text)
print('Official checkpoint bundle download complete.', flush=True)
