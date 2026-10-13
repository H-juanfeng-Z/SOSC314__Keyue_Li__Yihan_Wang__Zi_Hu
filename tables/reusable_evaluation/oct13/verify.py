import collections
import hashlib
import json
from pathlib import Path
P=Path(__file__).resolve().parent
read=lambda n:json.loads((P/n).read_text(encoding='utf8'))
m=read('manifest.json')
for n,h in m['files_sha256'].items():
    assert hashlib.sha256((P/n).read_bytes()).hexdigest()==h,n
q=read('question_index.json'); c=read('answer_scores.json'); f=read('fine_scores.json'); i=read('question_intents.json')
qs={r['question_id'] for r in q}
assert len(q)==len(qs)==len(c)==2000
assert {r['question_id'] for r in c}==qs
assert len(f)==len({(r['question_id'],r['scoring_version']) for r in f})==2000
assert len(i)==len({(r['question_id'],r['intent']) for r in i})==14000
assert set(collections.Counter(r['question_id'] for r in i).values())=={7}
assert all(r['question_id'] in qs for r in f+i)
cm={r['question_id']:r for r in c}
assert all(r['answer_id']==cm[r['question_id']]['answer_id'] for r in f)
assert all(r['score'] is None or (type(r['score']) is int and 0<=r['score']<=10) for r in f)
assert sum(r['status']=='valid' for r in c)==1992
fm=collections.defaultdict(list)
for r in f: fm[r['question_id']].append(r)
fine_ok={q for q,v in fm.items() if len(v)==2 and all(r['status']=='valid' and r['score'] is not None for r in v)}
intent_ok={q for q in qs if all(r['status']=='valid' and r['decision'] in ['yes','no'] for r in i if r['question_id']==q)}
coarse_ok={r['question_id'] for r in c if r['status']=='valid' and r['overall_numeric'] is not None}
assert len(fine_ok)==994 and len(intent_ok)==1987
assert len(coarse_ok & intent_ok)==1982
assert len(fine_ok & coarse_ok)==993
assert len(fine_ok & coarse_ok & intent_ok)==989
print('PASS: hashes, IDs, same-answer joins and valid denominators. Not an accuracy certification.')
