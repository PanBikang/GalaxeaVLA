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
    for key, array in arrays.items():
        assert np.isfinite(array).all(), key
    report = dict(checkpoint=checkpoint, task=task,
                  action_shapes={k: list(np.asarray(v).shape) for k, v in arrays.items()},
                  finite=True)
    (run_dir / 'inference_check.json').write_text(json.dumps(report, indent=2))
    np.savez(run_dir / 'inference_actions.npz', **arrays)
    print('Saved-checkpoint inference PASS:', report, flush=True)
finally:
    env.close()
