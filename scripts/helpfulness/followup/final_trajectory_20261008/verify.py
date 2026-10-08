"""Independent count audit, using raw metadata/ratings instead of batch tables."""
import json,collections,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent;B=R.parent
def read(p):return json.loads(p.read_text(encoding='utf8'))
ms=read(B/'answer_trajectory_20261007/answer_metadata.json')
rr=[json.loads(x) for x in (B/'answer_trajectory_20261007/results/calls.jsonl').read_text(encoding='utf8').splitlines()]
ix={r['pair_id']:r for r in rr};qs=collections.defaultdict(list)
for m in ms:qs[m['question_id']].append(m)
audit=[];summary=read(R/'results/cohorts.json')
for k in [2,3,5]:
 good=[];ties=0;providers=collections.Counter()
 for q,items in qs.items():
  items.sort(key=lambda m:m['answer_rank'])
  if len(items)<k:continue
  rs=[ix[m['pair_id']] for m in items[:k]]
  if any(r['status']!='valid' or r['parsed']['overall']=='uncertain' for r in rs):continue
  labs=[r['parsed']['overall']=='substantial' for r in rs];good.append(labs)
  if len(items)>k and items[k]['created_at']==items[k-1]['created_at']:ties+=1
  if k==2 and not labs[0] and labs[1]:providers[str(items[1]['self_answer'])]+=1
 original=next(x for x in summary if x['k']==k and x['model']=='14B' and x['threshold']=='substantial' and x['phase']=='pooled')
 counts=(len(good),sum(g[0] for g in good),sum(any(g) for g in good))
 assert counts==(original['complete_n'],original['first_positive'],original['by_rank_positive'][-1])
 audit.append(dict(k=k,n=counts[0],first=counts[1],any=counts[2],boundary_ties=ties,second_answer_providers=dict(providers),matched=True))
out={'independent_counts':audit,'limits':'Independent implementation of count audit, NOT independent semantic validation.'}
(R/'results/verification.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps(out,indent=2))
