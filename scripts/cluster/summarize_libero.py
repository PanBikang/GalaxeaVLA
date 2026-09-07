"""Require all 2,000 rollouts before comparing the release with paper Table 3."""
import argparse
import json
import math
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('run_dir', type=Path)
args = parser.parse_args()
paper = {'libero_spatial': 98.4, 'libero_object': 100.0, 'libero_goal': 98.6, 'libero_10': 98.6}
rows = []
for suite, claimed in paper.items():
    path = args.run_dir / suite / suite / f'{suite}_parallel_results.json'
    result = json.loads(path.read_text())
    tasks = result['tasks']
    assert len(tasks) == 10, (suite, 'expected 10 tasks', len(tasks))
    assert {int(t['task_id']) for t in tasks} == set(range(10)), suite
    assert all(t['total_episodes'] == 50 for t in tasks), (suite, 'incomplete trials')
    successes = sum(t['successes'] for t in tasks)
    assert 0 <= successes <= 500
    rows.append(dict(suite=suite, successes=successes, episodes=500,
                     success_rate_pct=successes/5, paper_pct=claimed,
                     difference_pp=successes/5-claimed))
successes = sum(row['successes'] for row in rows)
n = 2000
p = successes / n
z = 1.95996398454
center = (p + z*z/(2*n)) / (1 + z*z/n)
half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
report = dict(suites=rows, successes=successes, episodes=n, success_rate_pct=p*100,
              paper_pct=98.9, difference_pp=p*100-98.9,
              wilson_95_pct=[100*(center-half), 100*(center+half)],
              scope='Official released continuous-action LIBERO checkpoint; not a fresh 100K-step training or AR/CoT ablation.')
path = args.run_dir / 'paper_comparison.json'
path.write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
