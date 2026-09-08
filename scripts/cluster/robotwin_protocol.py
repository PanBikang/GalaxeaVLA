"""Freeze the paper task list and a balanced 32-worker evaluation plan."""
import argparse
import hashlib
import json
from pathlib import Path

from lxml import html
import yaml

parser=argparse.ArgumentParser()
parser.add_argument('--run-id',default='robotwin_20260908')
parser.add_argument('--workers',type=int,default=32)
args=parser.parse_args()
assert args.workers > 0 and args.run_id not in {'.','..'} and '/' not in args.run_id
root=Path(__file__).resolve().parents[2]
tree=html.fromstring((root/'runs/provenance/paper.html').read_text())
reference={}
for figure in tree.xpath('//figure[contains(@class,"ltx_table")]'):
    if 'Table 7' not in ' '.join(figure.xpath('.//figcaption//text()')):
        continue
    for row in figure.xpath('.//tr'):
        cells=[' '.join(' '.join(c.itertext()).split()) for c in row.xpath('./td|./th')]
        if len(cells)!=3 or cells[0] in {'Task','Average'}:
            continue
        reference['_'.join(cells[0].lower().split())]={'clean':float(cells[1]),'random':float(cells[2])}
limits=yaml.safe_load((root/'third_party/RoboTwin/task_config/_eval_step_limit.yml').read_text())
assert len(reference)==50 and set(reference)==set(limits), 'Paper task list does not match simulator'
cases=[]
for task in sorted(reference):
    for phase in ['clean','random']:
        cases.append({'task':task,'phase':phase,'task_config':'demo_clean' if phase=='clean' else 'demo_randomized',
                      'episodes':100,'step_limit':limits[task], 'paper_pct':reference[task][phase]})
loads=[0.]*args.workers
for case in sorted(cases,key=lambda c:c['step_limit']*(1.2 if c['phase']=='random' else 1),reverse=True):
    worker=min(range(args.workers),key=lambda i:loads[i])
    case['worker']=worker
    loads[worker]+=case['step_limit']*(1.2 if case['phase']=='random' else 1)
manifest={
    'run_id':args.run_id,'workers':args.workers,'cases':cases,'paper_reference':reference,
    'protocol':{'episodes_per_task_phase':100,'secondary_prefix_episodes':50,'seed':7,
                'instruction_type':'seen','replan_steps':24,'num_inference_steps':10,
                'vision_attention_backend':'sdpa','skip_get_obs_within_replan':True,'save_videos':False},
    'checkpoint':'checkpoints/g05-robotwin20/checkpoints/model_state_dict.pt',
    'checkpoint_sha256':'300d24c691f8d76846a218b9bd670312cf577b701ea5863f221af1b8d1c0d637',
    'hf_revision':'e312be81e90c56a55bcb26b57429bd39a335b449',
    'sources':json.loads((root/'runs/robotwin/provenance/sources.json').read_text()),
    'runtime_patch_sha256':hashlib.sha256((root/'scripts/cluster/robotwin_device.patch').read_bytes()).hexdigest(),
    'paper_protocol_note':'Main text specifies 100 trials per task/phase. Repo default is 50 and Table 7 uses 2-point increments; report the prespecified first-50 prefix separately.'}
out=root/'runs/robotwin'/args.run_id
out.mkdir(parents=True,exist_ok=True)
path=out/'manifest.json'
if path.exists() and json.loads(path.read_text())!=manifest:
    raise RuntimeError('Refusing to overwrite a different experiment manifest')
path.write_text(json.dumps(manifest,indent=2))
print('Frozen plan:',len(cases),'cases,',sum(c['episodes'] for c in cases),'episodes;',path,flush=True)
