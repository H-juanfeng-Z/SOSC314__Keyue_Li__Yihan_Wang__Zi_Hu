"""Audit frozen decision exports, not private raw completions or inference."""
import json,collections
from pathlib import Path
from protocol import validate
R=Path(__file__).resolve().parents[3];T=R/'tables/annotation_diagnostics/scale_repair_oct11'
def read(n):return json.loads((T/n).read_text(encoding='utf8'))
tasks={t['task_id']:t for t in read('tasks.json')};rows=read('outputs.json');a=read('analysis.json')
assert len(tasks)==len(rows)==len({r['task_id'] for r in rows})==242
assert set(tasks)=={r['task_id'] for r in rows}
old={str(r['question_id']):r for r in read('prior_fine_controls.json')}
oldids={q for q in old if not q.startswith('synthetic/')}
controls=[];groups=collections.defaultdict(list)
for r in rows:
 t=tasks[r['task_id']];e=validate(r['parsed'],t)
 assert e==r['errors'];assert r['status']==('invalid' if e else 'valid')
 if r['arm']=='fine_control':
  o=old[str(r['question_id'])];controls.append(r['parsed']==o['parsed'] and r['status']==o['status'])
 else:
  cohort='synthetic8' if r['suite']=='synthetic' else 'reviewed18' if str(r['question_id']) in oldids else 'new82'
  groups[cohort].append(r)
assert len(controls)==26 and all(controls)
assert dict(collections.Counter(r['status'] for r in rows))==a['statuses']
for cohort,rr in groups.items():
 for arm in ['coarse','fine']:
  z=[r for r in rr if r['arm']==arm];exp=a['stats'][cohort][arm]
  assert len(z)==exp['n']
  assert dict(collections.Counter(r['status'] for r in z))==exp['statuses']
  assert dict(collections.Counter(str(r['parsed']['score']) for r in z if r['status']=='valid'))==exp['scores']
 pairs=collections.defaultdict(dict)
 for r in rr:pairs[r['question_id']][r['arm']]=r
 valid=[p for p in pairs.values() if all(p[x]['status']=='valid' for x in ['coarse','fine'])]
 assert len(valid)==a['stats'][cohort]['paired_valid']
 assert dict(collections.Counter(str((p['coarse']['parsed']['score'],p['fine']['parsed']['score'])) for p in valid))==a['stats'][cohort]['coarse_fine_cells']
print(json.dumps({'passed':True,'tasks':242,'valid':236,'invalid':6,'controls_exact':26,'scope':'Export schema and summary consistency, not accuracy or private raw-log replay.'}))
