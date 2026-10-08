import json,argparse,collections
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();R=Path(__file__).resolve().parent
rows=[json.loads(s) for s in (a.root/'calls.jsonl').read_text(encoding='utf8').splitlines()];idx={r['pair_id']:r for r in rows};meta=json.loads((R/'answer_metadata.json').read_text(encoding='utf8'));groups=collections.defaultdict(list)
for m in meta:groups[m['question_id']].append(m)
details=[];author=collections.defaultdict(collections.Counter)
for q,ms in groups.items():
 ms.sort(key=lambda m:m['answer_rank']);valid=[]
 for m in ms:
  r=idx.get(m['pair_id'])
  if r and r['status']=='valid' and r['parsed']['overall']!='uncertain':
   valid.append((m,r['parsed']['overall']));author[str(m['self_answer'])][r['parsed']['overall']]+=1
 first=next((label for m,label in valid if m['answer_rank']==1),None);later=[label for m,label in valid if m['answer_rank']>1]
 substantial=[m for m,label in valid if label=='substantial'];details.append({'question_id':q,'planned_answers':len(ms),'archive_answer_count':ms[0]['answer_count'],'usable_ratings':len(valid),'complete_planned_trajectory':len(valid)==len(ms),'first':first,'later_valid_n':len(later),'any_substantial':bool(substantial) if valid else None,'later_substantial':any(x=='substantial' for x in later) if later else None,'first_observed_substantial_delay':min(m['delay_seconds'] for m in substantial) if substantial else None})
complete=[d for d in details if d['complete_planned_trajectory']];multi=[d for d in complete if d['planned_answers']>=2 and d['first'] is not None]
out={'records':len(rows),'unique_ids':len(idx),'statuses':dict(collections.Counter(r['status'] for r in rows)),'complete_trajectories':len(complete),'complete_multi_answer_questions':len(multi),'first_not_substantial_in_multi':sum(d['first']!='substantial' for d in multi),'later_substantial_after_first_not':sum(d['first']!='substantial' and d['later_substantial'] for d in multi),'author_groups':author,'questions':details,'limitations':'Selected reused development data, first5 only, independent answer ratings, no cumulative dialogue evaluation, no causal claims.'}
(a.root/'analysis.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps({k:v for k,v in out.items() if k!='questions'},indent=2))
