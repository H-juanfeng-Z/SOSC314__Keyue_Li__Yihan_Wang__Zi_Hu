import json,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent
read=lambda n:json.loads((R/n).read_text(encoding='utf8'))
m=read('manifest.json')
for n,h in m['files'].items():assert hashlib.sha256((R/n).read_bytes()).hexdigest()==h,n
a=read('scores_0_2_5000.json');b=read('scores_0_10_1000.json');c=read('scores_0_2_matched_1000.json')
am={r['question_id']:r for r in a}
assert len(a)==len(am)==5000
assert len(b)==len(c)==len({r['question_id'] for r in b})==1000
for x,y in zip(b,c):
    assert (x['question_id'],x['answer_id'])==(y['question_id'],y['answer_id'])
    assert y==am[x['question_id']]
    assert x['answer_rank']==y['answer_rank']==1
    for arm in ['format','scope']:
        z=x[arm];assert z['usable']==(z['status']=='valid' and z['score'] is not None)
        if z['usable']:assert type(z['score']) is int and 0<=z['score']<=10
for r in a:
    assert r['answer_rank']==1
    if r['usable']:assert r['status']=='valid' and r['overall_numeric'] in [0,1,2]
print('PASS: 5000 unique coarse questions; 1000 fine questions; exact paired coarse subset; hashes valid.')
