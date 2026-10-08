"""Incremental paired diagnostics; no model output is a gold label."""
import json,collections
from pathlib import Path
R=Path(__file__).resolve().parent;stages=json.loads((R/'stages.json').read_text());refs=json.loads((R/'references.json').read_text(encoding='utf8'));out={'stages':{},'comparisons':{}};data={}
for s in stages:
 p=R/s['name']/'results/calls.jsonl'
 if not p.exists():continue
 rs=[json.loads(x) for x in p.read_text(encoding='utf8').splitlines()];data[s['name']]=rs
 tasks={t['task_id']:t for t in json.loads((R/s['name']/'tasks.json').read_text(encoding='utf8'))};semantic=[]
 for r in rs:
  if r['status']!='valid':continue
  o=r['parsed'];positive=o.get('score',0) not in [0,None] if 'score' in o else o.get('overall') in ['partial','substantial']
  if positive and not o.get('evidence_ids',[]):semantic.append(r['task_id'])
 out['stages'][s['name']]={'planned':s['planned'],'records':len(rs),'unique_ids':len({r['task_id'] for r in rs}),'missing_tasks':s['planned']-len({r['task_id'] for r in rs}),'statuses':dict(collections.Counter(r['status'] for r in rs)),'positive_without_citation':semantic,'by_suite':{suite:dict(collections.Counter(str(r.get('parsed',{}).get('score',r.get('parsed',{}).get('overall',r.get('parsed',{}).get('decision')))) for r in rs if r['suite']==suite and r['status']=='valid')) for suite in sorted({r['suite'] for r in rs})}}
def compare(name,rs,old,key,lookup):
 ct=collections.Counter();n=0
 for r in rs:
  o=old.get(lookup(r))
  if o and r['status']==o['status']=='valid':ct[str(o['parsed'][key])+' -> '+str(r['parsed'][key])]+=1;n+=1
 out['comparisons'][name]={'paired_valid':n,'transitions':dict(ct)}
compare('14B_to_7B_intent',data.get('intent7',[]),refs['intent'],'decision',lambda r:r['pair_id']+'/'+r['suite'].removeprefix('intent_'))
compare('14B_to_7B_answers',data.get('answers7',[]),refs['answers'],'overall',lambda r:r['pair_id'])
feature14={r['task_id']:r for r in data.get('presentation14',[])}
for suite in sorted({r['suite'] for r in data.get('presentation7',[])}):compare(suite+'_14B_to_7B',[r for r in data['presentation7'] if r['suite']==suite],feature14,'score',lambda r:r['task_id'])
groups=collections.defaultdict(list)
for m in refs['metadata']:groups[m['base_pair_id']].append(m)
ct=collections.Counter();n=0
for r in data.get('collections14',[]):
 if r['status']!='valid':continue
 ms=sorted(groups[r['pair_id']],key=lambda m:m['answer_rank'])[:2 if r['arm']=='first2' else 5];labels=[refs['answers'].get(m['pair_id'],{}) for m in ms]
 if not all(x.get('status')=='valid' and x['parsed']['overall']!='uncertain' for x in labels):continue
 any_good=any(x['parsed']['overall']=='substantial' for x in labels);ct[r['arm']+'/single_any='+str(any_good)+' -> collection='+r['parsed']['overall']]+=1;n+=1
out['comparisons']['collection_vs_individual']={'paired_complete':n,'counts':dict(ct)}
out['limitations']='All reused development cases; shared-family agreement is not accuracy. Collections have greater token length and may be skipped; do not pool missing as negative. Features are not question quality.'
tmp=R/'analysis.tmp.json';tmp.write_text(json.dumps(out,indent=2),encoding='utf8');tmp.replace(R/'analysis.json')
print(json.dumps({k:{'records':v['records'],'planned':v['planned']} for k,v in out['stages'].items()}))
