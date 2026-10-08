import argparse,collections,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();R=Path(__file__).resolve().parent
rows=[json.loads(s) for s in (a.root/'calls.jsonl').read_text(encoding='utf8').splitlines()]
audit={c['pair_id']:c for c in json.loads((R/'case_audit.json').read_text())}
out={'records':len(rows),'unique_task_ids':len({r['task_id'] for r in rows}),'groups':{},'transitions':{},'limits':'Reused development data; no accuracy claims. Rubric changes are bundled.'}
for group in ['previous_selected48','additional_reused48']:
 for condition in ['original','empty','generic','swapped']:
  for arm in ['no_examples_control','concise_rubric']:
   rs=[r for r in rows if audit[r['pair_id']]['group']==group and r['suite']==condition and r['arm']==arm];v=[r for r in rs if r['status']=='valid']
   item={'attempted':len(rs),'valid':len(v),'overall':dict(collections.Counter(r['parsed']['overall'] for r in v))}
   if condition in ['empty','generic']:
    item['no_help_pass']=sum(r['parsed']['overall']=='unhelpful' and r['parsed']['actionability']==r['parsed']['sufficiency']==0 for r in v)
    item['all_zero_pass']=sum(r['parsed']['overall']=='unhelpful' and all(r['parsed'][k]==0 for k in ['relevance','actionability','sufficiency']) for r in v)
   out['groups'][group+'/'+condition+'/'+arm]=item
idx={(r['pair_id'],r['suite'],r['arm']):r for r in rows}
for group in ['previous_selected48','additional_reused48']:
 counts=collections.Counter()
 for pid,c in audit.items():
  if c['group']!=group:continue
  x=idx.get((pid,'original','no_examples_control'));y=idx.get((pid,'original','concise_rubric'))
  if x and y and x['status']==y['status']=='valid':counts[x['parsed']['overall']+' -> '+y['parsed']['overall']]+=1
 out['transitions'][group]=dict(counts)
prior=json.loads((R/'prior_controls.json').read_text());valid=same=0
for r in prior:
 x=idx.get((r['pair_id'],'original','no_examples_control'))
 if x and x['status']=='valid':
  valid+=1;same+=all(x['parsed'][k]==r['parsed'][k] for k in ['relevance','actionability','sufficiency','overall'])
out['control_reproduction']={'valid':valid,'same_scores':same}
(a.root/'analysis.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
