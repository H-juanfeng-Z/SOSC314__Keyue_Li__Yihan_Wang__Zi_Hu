import argparse,collections,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
rows=[json.loads(s) for s in (a.root/'calls.jsonl').read_text(encoding='utf8').splitlines()]
idx={(r['pair_id'],r['suite'],r['arm']):r for r in rows}
out={'records':len(rows),'conditions':{},'example_transitions':{},'limits':'Selected reused development cases; diagnostic constraints are not natural-data accuracy. Swapped answers are not guaranteed unrelated.'}
for condition in ['original','empty','generic','swapped']:
 for arm in ['original_examples','no_examples','balanced','balanced_reverse']:
  rs=[r for r in rows if r['suite']==condition and r['arm']==arm];valid=[r for r in rs if r['status']=='valid']
  out['conditions'][condition+'/'+arm]={'attempted':len(rs),'valid':len(valid),'overall':dict(collections.Counter(r['parsed']['overall'] for r in valid))}
  if condition in ['empty','generic']:out['conditions'][condition+'/'+arm]['strict_constraint_pass']=sum(r['parsed']['overall']=='unhelpful' and all(r['parsed'][d]==0 for d in ['relevance','actionability','sufficiency']) for r in valid)
 for arm in ['no_examples','balanced','balanced_reverse']:
  ct=collections.Counter()
  for pid in {r['pair_id'] for r in rows}:
   x=idx.get((pid,condition,'original_examples'));y=idx.get((pid,condition,arm))
   if x and y and x['status']==y['status']=='valid':ct[x['parsed']['overall']+' -> '+y['parsed']['overall']]+=1
  out['example_transitions'][condition+'/'+arm]=dict(ct)
ct=collections.Counter()
for pid in {r['pair_id'] for r in rows}:
 x=idx.get((pid,'original','balanced'));y=idx.get((pid,'original','balanced_reverse'))
 if x and y and x['status']==y['status']=='valid':ct[x['parsed']['overall']+' -> '+y['parsed']['overall']]+=1
out['balanced_order_transitions_original']=dict(ct)
audit={r['pair_id']:r for r in json.loads((Path(__file__).parent/'case_audit.json').read_text(encoding='utf8'))}
controls=[r for r in rows if r['suite']=='original' and r['arm']=='original_examples' and r['status']=='valid']
out['control_reproduction']={'valid':len(controls),'same_scores':sum(all(r['parsed'][k]==audit[r['pair_id']]['prior_control'][k] for k in ['relevance','actionability','sufficiency','overall']) for r in controls)}
(a.root/'analysis.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps(out,indent=2))
