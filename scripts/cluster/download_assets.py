"""Fetch just the official base and LIBERO bundles, pinned to one HF revision."""
import json
from pathlib import Path

import requests
from huggingface_hub import HfApi, snapshot_download

root = Path(__file__).resolve().parents[2]
out = root / 'runs' / 'provenance'
out.mkdir(parents=True, exist_ok=True)
api = HfApi()
revision = api.model_info('OpenGalaxea/G05').sha
(out / 'hf_revision.json').write_text(json.dumps({'repo': 'OpenGalaxea/G05', 'revision': revision}, indent=2))
print('Downloading OpenGalaxea/G05 revision', revision, flush=True)
snapshot_download('OpenGalaxea/G05', revision=revision, local_dir=root / 'checkpoints',
                  allow_patterns=['g05-base/*', 'g05-libero/*', 'action_tokenizer.pt',
                                  'qwen3_5_2b_base_processor/*'], max_workers=4)
for name, target in [('action_tokenizer.pt', '../action_tokenizer.pt'),
                     ('hf_processor', '../qwen3_5_2b_base_processor')]:
    link = root / 'checkpoints' / 'g05-libero' / name
    if not link.exists() and not link.is_symlink():
        link.symlink_to(target)
for path in ['g05-base/checkpoints/model_state_dict.pt', 'g05-libero/model.pt',
             'action_tokenizer.pt', 'qwen3_5_2b_base_processor/tokenizer_config.json']:
    assert (root / 'checkpoints' / path).is_file(), path
response = requests.get('https://arxiv.org/html/2608.11739v1', timeout=60)
response.raise_for_status()
(out / 'paper.html').write_text(response.text)
print('Official checkpoint bundles and paper saved.', flush=True)
