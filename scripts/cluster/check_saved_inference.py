"""Load the saved training bundle and infer actions from a real simulator observation."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

from experiments.libero.libero_eval_utils import (
    build_libero_raw_obs, ensure_libero_import, get_libero_env, load_libero_init_states,
)
from g05.models.g05.inferencer import PolicyInferencer
from g05.utils.checkpoint.ckpt_utils import find_run_dir, load_config_from_run_dir
from scripts.serve_policy import setup, build_obs_dict

checkpoint = str(Path(sys.argv[1]).resolve())
run_dir = find_run_dir(checkpoint)
cfg = load_config_from_run_dir(run_dir, checkpoint, ['eval_embodiment=libero'])
# Verify that the training bundle contains a real parameter update, not just a
# successful serialization of the initialization checkpoint.
saved = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
initial = torch.load(cfg.model.pretrained_ckpt, map_location='cpu', mmap=True, weights_only=False)
initial_state = initial.get('model_state_dict', initial.get('state_dict', initial))
update = None
for key, value in saved['model_state_dict'].items():
    before = initial_state.get(key)
    if ('tokenizer' in key or not key.endswith('weight') or not isinstance(value, torch.Tensor)
            or value.numel() == 0 or not isinstance(before, torch.Tensor)
            or value.shape != before.shape or value.dtype != before.dtype or not value.is_floating_point()):
        continue
    delta = (value.reshape(-1)[:4096].float() - before.reshape(-1)[:4096].float()).abs().max().item()
    if np.isfinite(delta) and delta > 0:
        update = {'key': key, 'sample_max_abs_delta': delta}
        break
assert update is not None, 'No parameter update found relative to initialization'
training_step = int(saved['step'])
assert training_step > 0
del saved, initial, initial_state, value, before
policy, processor = setup(cfg)
ensure_libero_import()
from libero.libero import benchmark

suite = benchmark.get_benchmark_dict()['libero_10']()
env, task = get_libero_env(suite.get_task(0), resolution=256, seed=7)
try:
    env.reset()
    obs = env.set_init_state(load_libero_init_states(suite, 0)[0])
    for _ in range(20):
        obs, _, _, _ = env.step([0.] * 6 + [-1.])
    raw, _ = build_libero_raw_obs(obs, task, embodiment_type='libero')
    raw['frequency'] = 20.
    with torch.inference_mode():
        action = PolicyInferencer(policy, processor).infer_one(build_obs_dict(raw, processor))
    arrays = {k: v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v
              for k, v in action.items() if not k.startswith('_')}
    assert arrays, 'No actions returned'
    for key, dimension in [('right_ee_pose', 6), ('right_gripper', 1)]:
        assert key in arrays, f'Missing LIBERO action group: {key}'
        assert tuple(np.asarray(arrays[key]).shape[-2:]) == (32, dimension), (key, np.asarray(arrays[key]).shape)
    for key, array in arrays.items():
        assert np.isfinite(array).all(), key
    report = dict(checkpoint=checkpoint, task=task,
                  training_step=training_step, parameter_update=update,
                  action_shapes={k: list(np.asarray(v).shape) for k, v in arrays.items()},
                  absent_groups=sorted(action.get('_absent_keys', [])),
                  finite=True)
    (run_dir / 'inference_check.json').write_text(json.dumps(report, indent=2))
    np.savez(run_dir / 'inference_actions.npz', **arrays)
    print('Saved-checkpoint inference PASS:', report, flush=True)
finally:
    env.close()
