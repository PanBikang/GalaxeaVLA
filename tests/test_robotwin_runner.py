import ast
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.robotwin.eval_robotwin_single import (
    _collect_robotwin_result, _patch_robotwin_eval_policy_source, _select_visible_gpu,
)


@pytest.mark.parametrize('index,mask,expected', [(0,'4','4'), (1,'2,6','6'), (0,'GPU-abc','GPU-abc'), (2,None,'2')])
def test_gpu_selection_respects_the_allocated_mask(index, mask, expected):
    assert _select_visible_gpu(index, mask) == expected


@pytest.mark.parametrize('index,mask', [(1,'4'), (0,''), (-1,'0')])
def test_invalid_gpu_selection_fails(index, mask):
    with pytest.raises(ValueError):
        _select_visible_gpu(index, mask)


def test_result_requires_current_run_and_exact_episode_count(tmp_path):
    dest = tmp_path / 'result'
    phase = dest / 'demo_clean'
    phase.mkdir(parents=True)
    (phase / '_result.txt').write_text('Timestamp: test\n\n0.5')
    rows = [{'trial': 0, 'seed': 100, 'success': True}, {'trial': 1, 'seed': 101, 'success': False}]
    (phase / 'episodes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    _collect_robotwin_result(tmp_path,'task','policy','demo_clean',dest,2)
    assert (dest / '_result_clean.txt').is_file()
    with pytest.raises(ValueError):
        _collect_robotwin_result(tmp_path,'task','policy','demo_clean',dest,100)
    with pytest.raises(FileNotFoundError):
        _collect_robotwin_result(tmp_path,'task','policy','demo_randomized',dest,2)


def test_patched_loop_resumes_without_replaying_completed_trials(tmp_path):
    source_path = Path(__file__).resolve().parents[1] / 'third_party/RoboTwin/script/eval_policy.py'
    if not source_path.exists():
        pytest.skip('Pinned RoboTwin source is required for the adapter integration check')
    patched = _patch_robotwin_eval_policy_source(source_path.read_text())
    assert _patch_robotwin_eval_policy_source(patched) == patched
    tree = ast.parse(patched)
    fn = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name == 'eval_policy')

    class Env:
        def __init__(self): self.policy_seeds=[]
        def setup_demo(self, seed, **kwargs):
            self.seed=seed; self.plan_success=True; self.eval_success=False
            self.eval_video_path=None; self.render_freq=0; self.step_lim=3; self.take_action_cnt=0
        def play_once(self): return {'info': {}}
        def check_success(self): return True
        def close_env(self, **kwargs): pass
        def set_instruction(self, instruction): pass
        def get_obs(self): return {}

    def step(env, model, observation):
        env.policy_seeds.append(env.seed)
        env.take_action_cnt += 1
        env.eval_success=True

    scope = {'Path':Path,'json':json,'np':np,'UnStableError':type('UnStableError',(Exception,),{}),
             'generate_episode_descriptions':lambda *a:[{'seen':['task']}],
             'eval_function_decorator':lambda policy,name:step if name=='eval' else lambda model:None}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'<RoboTwin adapter test>','exec'),scope)
    args = {'task_name':'task','policy_name':'policy','task_config':'demo_clean','ckpt_setting':'ckpt',
            'clear_cache_freq':5,'render_freq':0,'_g05_skip_obs':False,
            '_g05_episode_log':str(tmp_path/'episodes.jsonl')}
    env=Env()
    scope['eval_policy']('task',env,args,object(),100000,test_num=2,instruction_type='seen')
    scope['eval_policy']('task',env,args,object(),100000,test_num=3,instruction_type='seen')
    rows=[json.loads(s) for s in (tmp_path/'episodes.jsonl').read_text().splitlines()]
    assert env.policy_seeds == [100000,100001,100002]
    assert [r['trial'] for r in rows] == [0,1,2]
    assert env.suc == env.test_num == 3
