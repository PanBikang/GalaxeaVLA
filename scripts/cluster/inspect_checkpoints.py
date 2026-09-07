"""Record published configuration choices and tensor metadata for every G05 bundle."""
import gc
import json
from collections import Counter
from pathlib import Path

import torch
import yaml

root = Path(__file__).resolve().parents[2]
records = []
for bundle in sorted((root / 'checkpoints').glob('g05-*')):
    cfg = yaml.safe_load((bundle / '.hydra/config.yaml').read_text())
    arch = cfg['model']['model_arch']
    weights = bundle / 'model.pt'
    if not weights.exists():
        weights = bundle / 'checkpoints/model_state_dict.pt'
    # These are the authenticated official release files. mmap avoids copying
    # all five 11 GB bundles into RAM just to inspect tensor metadata.
    checkpoint = torch.load(weights, map_location='cpu', mmap=True, weights_only=False)
    state = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
    tensors = {k: v for k, v in state.items() if isinstance(v, torch.Tensor)}
    if not tensors:
        raise ValueError(f'Unrecognized checkpoint layout: {weights}: {list(checkpoint)[:10]}')
    record = {
        'bundle': bundle.name, 'path': str(weights.relative_to(root)),
        'bytes': weights.stat().st_size,
        'tensor_count': len(tensors), 'tensor_elements': sum(v.numel() for v in tensors.values()),
        'tensor_dtypes': dict(Counter(str(v.dtype) for v in tensors.values())),
        'policy_class': arch.get('_target_'), 'action_dim': arch.get('action_dim'),
        'horizon_steps': arch.get('horizon_steps'),
        'discrete_action': arch.get('discrete_action'),
        'continuous_action': arch.get('continuous_action'),
        'predict_cot': arch.get('predict_cot'),
        'processor': arch.get('hf_processor_path'),
        'stats_exists': (bundle / 'dataset_stats.json').is_file(),
        'action_expert_layer_types': sorted(set(arch.get('action_expert', {}).get('layer_types', []))),
    }
    records.append(record)
    print(json.dumps(record), flush=True)
    del checkpoint, state, tensors
    gc.collect()
out = root / 'runs/provenance/checkpoint_inspection.json'
out.write_text(json.dumps(records, indent=2))
print('All checkpoint tensor layouts inspected:', out, flush=True)
