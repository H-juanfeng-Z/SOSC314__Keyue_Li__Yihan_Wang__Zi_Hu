"""Frozen finite analysis batch, question-level pairing, no GPU or uploads."""
import collections,hashlib,json,random,time,math
from pathlib import Path
R=Path(__file__).resolve().parent;B=R.parent;OUT=R/'results';OUT.mkdir(exist_ok=True)
def read(p):return json.loads(p.read_text(encoding='utf8'))
def rows(p):return [json.loads(x) for x in p.read_text(encoding='utf8').splitlines()]
def save(name,obj):
    p=OUT/name;tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf8');tmp.replace(p)
def event(stage,**kw):
    obj=dict(time=time.time(),stage=stage,**kw)
    with (OUT/'events.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(obj)+'\n')
    save('progress.json',obj);print(json.dumps(obj),flush=True)
def bootstrap(values,key):
    if not values:return None
    rng=random.Random(int(hashlib.sha256(key.encode()).hexdigest()[:12],16));n=len(values)
    dist=sorted(sum(values[rng.randrange(n)] for _ in range(n))/n for _ in range(2000))
    return [dist[49],dist[1949]]
started=time.time();assert not (OUT/'summary.json').exists(),'Refuse overwriting completed batch'
paths={'metadata':B/'answer_trajectory_20261007/answer_metadata.json','14B':B/'answer_trajectory_20261007/results/calls.jsonl','7B':B/'integrated_20261008/answers7/results/calls.jsonl','intent':B/'association_sensitivity_20261007/input.json'}
meta=read(paths['metadata']);pool=read(paths['intent']);qs={p['question_id']:p for p in pool};models={s:rows(paths[s]) for s in ['14B','7B']}
assert len(qs)==390 and len(meta)==715
assert len({m['answer_id'] for m in meta})==len(meta)
groups=collections.defaultdict(list)
for m in meta:groups[m['question_id']].append(m)
assert set(groups)==set(qs)
for q,ms in groups.items():
    ms.sort(key=lambda m:m['answer_rank']);assert [m['answer_rank'] for m in ms]==list(range(1,len(ms)+1))
    assert all(m['delay_seconds']>=0 for m in ms)
    assert all(ms[i]['delay_seconds']<=ms[i+1]['delay_seconds'] for i in range(len(ms)-1))
index={}
for model,rr in models.items():
    assert len(rr)==len({x['pair_id'] for x in rr})==715
    index[model]={x['pair_id']:x for x in rr};assert set(index[model])=={m['pair_id'] for m in meta}
def label(model,m):
    r=index[model][m['pair_id']]
    return r['parsed']['overall'] if r['status']=='valid' and r['parsed']['overall']!='uncertain' else None
save('manifest.json',dict(inputs={k:dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for k,p in paths.items()},script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),reserved_used=False,bootstrap_draws=2000,main='14B, substantial, first2, pooled complete cases',secondary='first3/first5, any_help threshold, phases, self answers, tie exclusion, fixed-window rates,7B common cases, intent subgroup',limits=['Selected reused answered development questions','At least k saved answers is a selected subgroup','Any-help-with-more-answers monotonicity is mathematical, not causal','Only first5 saved,12 questions truncated','Unknown labels not coded negative','Same-family model agreement not accuracy','Bootstrap conditional on sample and fixed labels; does not capture measurement bias','Edited snapshots cannot establish temporal causality','No censoring endpoint: descriptive times only','Exploratory subgroups overlap; no multiplicity-adjusted causal inference']))
event('integrity_complete',questions=390,answers=715)
stats=[];all_data={}
for model in ['14B','7B']:
 for k in [2,3,5]:
  for threshold,positive in [('substantial',{'substantial'}),('any_help',{'partial','substantial'})]:
   for phase in ['pooled','primary','extension']:
    cand=[q for q in groups if len(groups[q])>=k and (phase=='pooled' or qs[q]['phase']==phase)]
    complete=[];bounds=[]
    for q in cand:
     ms=groups[q][:k];ls=[label(model,m) for m in ms];bs=[None if x is None else int(x in positive) for x in ls]
     bounds.append((int(any(x==1 for x in bs)),int(any(x is None or x==1 for x in bs))))
     if None in bs:continue
     ties=len({m['created_at'] for m in ms})!=len(ms)
     complete.append(dict(question_id=q,first=bs[0],cumulative=[int(any(bs[:j])) for j in range(1,k+1)],rescue=int(not bs[0] and any(bs[1:])),labels=ls,positive=bs,ties=ties))
    n=len(complete);key=f'{model}/{k}/{threshold}/{phase}';all_data[key]=complete
    rescue=[r['rescue'] for r in complete];at_risk=[r['rescue'] for r in complete if not r['first']]
    st=dict(model=model,k=k,threshold=threshold,phase=phase,candidate_n=len(cand),complete_n=n,excluded_invalid_n=len(cand)-n,first_positive=sum(r['first'] for r in complete),by_rank_positive=[sum(r['cumulative'][j] for r in complete) for j in range(k)],rescue_n=sum(rescue),first_negative_n=len(at_risk),increment=sum(rescue)/n if n else None,increment_bootstrap95=bootstrap(rescue,key),conditional_rescue_rate=sum(at_risk)/len(at_risk) if at_risk else None,conditional_rescue_bootstrap95=bootstrap(at_risk,key+'/risk'),all_candidate_any_positive_bounds=[sum(x[j] for x in bounds)/len(bounds) for j in [0,1]] if bounds else None,tie_questions=sum(r['ties'] for r in complete))
    stats.append(st)
 event('cohort_model_complete',model=model,cells=len(stats));save('cohorts.json',stats)
event('cohorts_complete')
author=[];timing=[];intent=[];cross=[]
categories=['api_usage','discrepancy','errors','review','conceptual','api_change','learning']
for k in [2,3,5]:
 d=all_data[f'14B/{k}/substantial/pooled'];known=[r for r in d if all(m['self_answer'] is not None for m in groups[r['question_id']][:k])]
 rescue_types=collections.Counter();examples=[]
 for r in d:
  if not r['rescue']:continue
  ms=groups[r['question_id']][:k];good=[m for m,b in zip(ms,r['positive']) if b and m['answer_rank']>1];earliest=min(m['delay_seconds'] for m in good);firstgood=[m for m in good if m['delay_seconds']==earliest]
  who={str(m['self_answer']) for m in firstgood};kind=next(iter(who)) if len(who)==1 else 'mixed_same_timestamp';rescue_types[kind]+=1
  examples.append(dict(question_id=r['question_id'],labels=r['labels'],first_positive_answer_ids=[m['answer_id'] for m in firstgood],earliest_provider=kind,delay_seconds=earliest))
 author.append(dict(k=k,complete_n=len(d),all_authors_known_n=len(known),unknown_author_questions=len(d)-len(known),known_any_positive=sum(any(r['positive']) for r in known),known_other_positive=sum(any(b and m['self_answer'] is False for b,m in zip(r['positive'],groups[r['question_id']][:k])) for r in known),no_other_answer_in_prefix=sum(all(m['self_answer'] for m in groups[r['question_id']][:k]) for r in known),rescues_by_earliest_provider=dict(rescue_types),examples=examples))
 for hours in [24,168,720]:
  within=sum(any(b and m['delay_seconds']<=hours*3600 for b,m in zip(r['positive'],groups[r['question_id']][:k])) for r in d)
  timing.append(dict(k=k,hours=hours,n=len(d),observed_positive_by_window=within,warning='Absence within window is not failure; no validated archive cutoff. Only selected questions and saved first k answers.'))
 a={r['question_id']:r for r in d};b={r['question_id']:r for r in all_data[f'7B/{k}/substantial/pooled']};common=sorted(set(a)&set(b));ct=collections.Counter()
 for q in common:ct[f"{a[q]['rescue']} -> {b[q]['rescue']}"]+=1
 cross.append(dict(k=k,valid14=len(a),valid7=len(b),common_n=len(common),rescue_transitions=dict(ct),coverage_limit='Common-valid results may be selected by7B formatting failures.'))
 for cat in categories:
  for arm in ['direct','question_first']:
   bins={'yes':[],'no':[]}
   for r in d:
    z=qs[r['question_id']]['intent'][cat+'/'+arm]
    if z['status']=='valid' and z['value'] in bins:bins[z['value']].append(r)
   record=dict(k=k,category=cat,arm=arm,groups={v:dict(n=len(rr),first_positive=sum(x['first'] for x in rr),any_positive=sum(x['cumulative'][-1] for x in rr),rescue=sum(x['rescue'] for x in rr),first_negative=sum(not x['first'] for x in rr)) for v,rr in bins.items()})
   record['suppress_between_group_estimate']=min(map(len,bins.values()))<10
   record['first_negative_small']=any(sum(not x['first'] for x in rr)<10 for rr in bins.values())
   intent.append(record)
save('authors.json',author);save('observed_time_windows.json',timing);save('intent_groups.json',intent);save('cross_model.json',cross);save('question_level.json',all_data)
event('author_intent_sensitivity_complete')
ties=[]
for k in [2,3,5]:
 d=all_data[f'14B/{k}/substantial/pooled'];noties=[r for r in d if not r['ties']]
 ties.append(dict(k=k,original_n=len(d),no_ties_n=len(noties),no_ties_rescue=sum(r['rescue'] for r in noties)))
save('tie_sensitivity.json',ties)
summary=dict(completed=True,elapsed_seconds=time.time()-started,cohort_cells=len(stats),intent_cells=len(intent),main=next(x for x in stats if x['model']=='14B' and x['k']==2 and x['threshold']=='substantial' and x['phase']=='pooled'),next_decision='Do not claim validated quality or causal effects. Use fixed-prefix main result; discuss sparse intent rescue groups and independent semantic validation before further expansion.')
save('summary.json',summary);event('finished',elapsed_seconds=summary['elapsed_seconds']);print(json.dumps(summary,indent=2))
