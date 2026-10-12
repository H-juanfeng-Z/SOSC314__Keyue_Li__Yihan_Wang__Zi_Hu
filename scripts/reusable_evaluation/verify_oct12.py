import collections,hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parents[2]
read=lambda p:json.loads(p.read_text(encoding='utf8'))
T=R/'tables/reusable_evaluation/oct12'; D=R/'tables/annotation_diagnostics/fine_expansion_oct12'
for d in [T,D]:
 for n,h in read(d/'manifest.json')['sha256'].items():assert hashlib.sha256((d/n).read_bytes()).hexdigest()==h,n
s=read(T/'answer_scores.json'); a=read(T/'question_intents.json'); summary=read(T/'summary.json')
assert len(s)==1820==len({x['answer_id'] for x in s})
assert len(a)==1000==len({x['question_id'] for x in a})
assert dict(collections.Counter(x['status'] for x in s))==summary['answer_statuses']
for x in a:assert x['complete_valid']==(len(x['decisions'])==7 and all(v['status']=='valid' and v['decision'] in ['yes','no'] for v in x['decisions'].values()))
assert sum(x['complete_valid'] for x in a)==997
first={x['question_id']:x for x in s if x['answer_rank']==1};assert len(first)==1000
j=[x for x in a if x['complete_valid'] and first[x['question_id']]['status']=='valid'];assert len(j)==995
assert dict(collections.Counter(first[x['question_id']]['scores']['overall'] for x in j))==summary['paired_distribution']
for g,v in summary['overlapping_intent_groups'].items():
 selected=[x for x in j if x['decisions'][g]['decision']=='yes']
 assert len(selected)==v['n']
 assert dict(collections.Counter(first[x['question_id']]['scores']['overall'] for x in selected))==v['distribution']
rows=read(D/'paired_intent_scores.json');assert len(rows)==398==len({x['question_id'] for x in rows})
assert len(read(D/'paired_scores.json'))==400
for g,v in read(D/'statistics.json')['stats']['groups'].items():
 selected=[x for x in rows if g in x['categories']];assert len(selected)==v['n']
 for arm in ['format','scope']:assert abs(sum(x[arm] for x in selected)/len(selected)-v[arm]['mean'])<1e-12
assert not read(D/'joint_family_check.json')['excluding_zero']
print('PASS: manifests,1820answers,1000questions,997complete,995paired,group counts and398candidate means. Does not validate semantic accuracy.')
