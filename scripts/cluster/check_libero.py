"""Check standard LIBERO initial states, rendering, and local data schema."""
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from experiments.libero.libero_eval_utils import ensure_libero_import, get_libero_env, load_libero_init_states

ensure_libero_import()
from libero.libero import benchmark

records = []
for suite_name in ['libero_spatial', 'libero_object', 'libero_goal', 'libero_10']:
    suite = benchmark.get_benchmark_dict()[suite_name]()
    assert suite.n_tasks == 10
    for task_id in range(suite.n_tasks):
        assert len(load_libero_init_states(suite, task_id)) >= 50
    task = suite.get_task(0)
    env, description = get_libero_env(task, resolution=256, seed=7)
    try:
        env.reset()
        obs = env.set_init_state(load_libero_init_states(suite, 0)[0])
        for _ in range(20):
            obs, reward, done, info = env.step([0.] * 6 + [-1.])
        assert obs['agentview_image'].shape == (256, 256, 3)
        assert np.isfinite(obs['robot0_eef_pos']).all()
    finally:
        env.close()
    root = Path(os.environ['LIBERO_DATA_ROOT']) / f'{suite_name}_no_noops_lerobot'
    metadata = json.loads((root / 'meta/info.json').read_text())
    assert metadata['codebase_version'] == 'v2.1'
    for key in ['observation.images.image', 'observation.images.wrist_image']:
        assert metadata['features'][key]['shape'] == [512, 512, 3]
    table = pq.read_table(next(root.glob('data/chunk-*/*.parquet')))
    assert np.asarray(table['action'][0].as_py()).shape == (7,)
    assert np.asarray(table['observation.state'][0].as_py()).shape == (8,)
    record = dict(suite=suite_name, renderer=os.environ.get('MUJOCO_GL'), tasks=suite.n_tasks, episodes=metadata['total_episodes'],
                  frames=metadata['total_frames'], render='PASS', source=str(root))
    records.append(record)
    print(record, flush=True)
out = Path('runs/provenance/libero_check.json')
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(records, indent=2))
print('LIBERO standard simulator and local dataset schema PASS', flush=True)
