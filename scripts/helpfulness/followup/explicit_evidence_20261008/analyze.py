"""Read-only measurement diagnostics. Emits JSON to stdout, never repairs labels."""
import collections,json,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent;OLD=R.parent/'integrated_20261008'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def rows(p):return [json.loads(x) for x in p.read_text(encoding='utf8').splitlines()]
expected=read(R/'expectations.json');out={'stages':{},'limits':read(R/'plan.json')['limits']}
for s in read(R/'stages.json'):
    p=R/s['name'];f=p/'results/calls.jsonl'
    if not f.exists():continue
    rs=rows(f);ts=read(p/'tasks.json');ids={t['task_id'] for t in ts};task={t['task_id']:t for t in ts};ri=[r['task_id'] for r in rs]
    d={'planned':len(ts),'records':len(rs),'missing':len(ids-set(ri)),'extra':len(set(ri)-ids),'duplicates':len(ri)-len(set(ri)),'statuses':dict(collections.Counter(r['status'] for r in rs)),'input_hash_matches':hashlib.sha256((p/'tasks.json').read_bytes()).hexdigest()==read(p/'results/manifest.json')['input_sha256']}
    groups=collections.defaultdict(list)
    for r in rs:groups[r['suite']+'/'+r['arm']+'/'+('synthetic' if r['pair_id'].startswith('SYN-') else 'natural')].append(r)
    d['groups']={}
    for k,rr in groups.items():
        v=[r for r in rr if r['status']=='valid'];rec={'attempted':len(rr),'valid':len(v),'labels':dict(collections.Counter(str(r['parsed'].get('score',r['parsed'].get('overall'))) for r in v))}
        if k.endswith('/synthetic'):rec.update(designed_expected_matches=sum(r['parsed']['score']==expected[r['task_id']] for r in v),invalid_not_counted_correct=len(rr)-len(v))
        d['groups'][k]=rec
    lookup={(r['pair_id'],r['suite'],r['arm']):r for r in rs};trans=collections.defaultdict(collections.Counter);bad=collections.defaultdict(collections.Counter)
    for r in rs:
        if r['status']!='valid':
            bad[r['arm']][r['status']]+=1
            if r.get('parsed'):
                e=r['parsed'].get('evidence_ids');allowed=task[r['task_id']]['lines']
                if isinstance(e,list):
                    bad[r['arm']]['over_three']+=len(e)>3
                    bad[r['arm']]['nonexistent_or_nonstr_ID']+=any(not isinstance(x,str) or x not in allowed for x in e)
        if r['arm']=='baseline':continue
        a=lookup.get((r['pair_id'],r['suite'],'baseline'))
        if not a or a['status']!='valid' or r['status']!='valid':continue
        key='score' if 'score' in r['parsed'] else 'overall'
        kind='synthetic' if r['pair_id'].startswith('SYN-') else 'natural'
        trans[r['suite']+'/'+kind][str(a['parsed'][key])+' -> '+str(r['parsed'][key])]+=1
    d['paired_transitions']={k:dict(v) for k,v in trans.items()};d['failure_breakdown']={k:dict(v) for k,v in bad.items()}
    oldname={'features14':'presentation14','features7':'presentation7','format7':'answers7'}[s['name']]
    historical={(r['pair_id'],r['suite']):r for r in rows(OLD/oldname/'results/calls.jsonl')}
    control=collections.Counter()
    for r in rs:
        if r['arm']!='baseline' or r['pair_id'].startswith('SYN-'):continue
        old=historical[(r['pair_id'],r['suite'])];control['cases']+=1;control['identical_messages']+=r['messages']==old['messages'];control['same_raw']+=r.get('raw')==old.get('raw');control['same_status']+=r['status']==old['status']
    d['archived_control']=dict(control);out['stages'][s['name']]=d
print(json.dumps(out,indent=2))
