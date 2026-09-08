"""Run one GPU's disjoint task/phase shard, retaining per-episode progress."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from experiments.robotwin.eval_robotwin_single import _collect_robotwin_result

parser=argparse.ArgumentParser()
parser.add_argument('manifest',type=Path)
parser.add_argument('--base-worker',type=int,required=True)
args=parser.parse_args()
manifest=json.loads(args.manifest.read_text())
worker=args.base_worker+int(os.environ.get('SLURM_PROCID','0'))
mask=os.environ.get('CUDA_VISIBLE_DEVICES','')
if not mask or len(mask.split(','))!=1:
    raise RuntimeError(f'Worker requires exactly one Slurm-visible GPU; received {mask!r}')
root=Path(__file__).resolve().parents[2]
run_id=manifest['run_id']
out=root/'evaluate_results/robotwin/model_state_dict'/run_id
record_dir=args.manifest.parent/'workers'
record_dir.mkdir(parents=True,exist_ok=True)
record_file=record_dir/f'worker_{worker:02d}.json'
cases=[case for case in manifest['cases'] if case['worker']==worker]
print(f'Worker {worker}, host={os.uname().nodename}, GPU={mask}, cases={len(cases)}',flush=True)
records=[]
for case in cases:
    task,phase=case['task'],case['task_config']
    destination=out/task
    if (destination/phase/'_result.txt').exists():
        _collect_robotwin_result(root/'third_party/RoboTwin',task,'galaxeafm_policy',phase,destination,case['episodes'])
        records.append({**case,'status':'already_complete'})
        continue
    cmd=[sys.executable,'-u','experiments/robotwin/eval_robotwin_single.py','task=robotwin',
         f'ckpt={manifest["checkpoint"]}','gpu_id=0',
         f'EVALUATION.robotwin_root={root/"third_party/RoboTwin"}',
         f'EVALUATION.task_name={task}',f'EVALUATION.task_config={phase}',
         f'EVALUATION.eval_num_episodes={case["episodes"]}',
         f'EVALUATION.output_dir=runs/robotwin/{run_id}',
         'seed=7','EVALUATION.replan_steps=24','EVALUATION.num_inference_steps=10',
         'EVALUATION.vision_attention_backend=sdpa','EVALUATION.save_videos=false',
         'EVALUATION.skip_get_obs_within_replan=true']
    print('START',task,phase,flush=True)
    result=subprocess.run(cmd,cwd=root)
    records.append({**case,'status':'complete' if result.returncode==0 else 'failed',
                    'return_code':result.returncode})
    temporary=record_file.with_suffix('.tmp')
    temporary.write_text(json.dumps({'worker':worker,'slurm_job':os.environ['SLURM_JOB_ID'],
                                     'gpu':mask,'records':records},indent=2))
    temporary.replace(record_file)
    print('END',task,phase,'exit=',result.returncode,flush=True)
if any(r['status']=='failed' for r in records):
    raise SystemExit(1)
