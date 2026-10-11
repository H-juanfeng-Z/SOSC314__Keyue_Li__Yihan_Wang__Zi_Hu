"""Create frozen400-question shards from existing2000/400 inputs; no inference."""
import argparse,json,hashlib,copy
from pathlib import Path
p=argparse.ArgumentParser()
for k in ['questions','previous-tasks','out']:p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();qs=json.loads(a.questions.read_text(encoding='utf8'));old=json.loads(a.previous_tasks.read_text(encoding='utf8'))
oldids={t['question_id'] for t in old};templates={t['suite']:t for t in old}
assert len(qs)==len({q['question_id'] for q in qs})==2000
assert len(oldids)==400 and len(templates)==7
assert oldids<={q['question_id'] for q in qs}
new=sorted([q for q in qs if q['question_id'] not in oldids],key=lambda q:hashlib.sha256(f"{q['question_id']}:intent-completion:314".encode()).hexdigest())
assert len(new)==1600;controlids=sorted(oldids)[:2]
a.out.mkdir(parents=True,exist_ok=False)
for s in range(4):
 tasks=[]
 for t in old:
  if t['question_id'] in controlids:
   t=copy.deepcopy(t);t['task_id']=f'control/shard{s}/'+t['task_id'];tasks.append(t)
 for q in new[s::4]:
  lines={'T':q['title'],**{f'Q{i+1}':v for i,v in enumerate(q['body_text'].splitlines()) if v.strip()}}
  for suite,t in sorted(templates.items()):
   prefix=t['messages'][1]['content'].split('QUESTION LINES:\n')[0]
   tasks.append(dict(task_id=f"{suite}/{q['base_pair_id']}/direct",pair_id=q['base_pair_id'],question_id=q['question_id'],suite=suite,arm='direct',validator='intent',lines=lines,messages=[t['messages'][0],dict(role='user',content=prefix+'QUESTION LINES:\n'+json.dumps(lines,ensure_ascii=False))]))
 assert len(tasks)==len({t['task_id'] for t in tasks})==2814
 d=a.out/f'shard{s}';d.mkdir();(d/'tasks.json').write_text(json.dumps(tasks,ensure_ascii=False,indent=2),encoding='utf8')
print('Prepared11256 tasks; no server operation performed.')
