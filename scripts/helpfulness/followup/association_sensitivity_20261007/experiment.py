"""Exploratory intent/helpfulness association, with paired measurement sensitivity.
No gold labels, causal effects, or held-out evaluation. Standard library only.
"""
import argparse,collections,hashlib,json,random,time
from pathlib import Path
R=Path(__file__).resolve().parent
def readlines(p): return [json.loads(s) for s in p.read_text(encoding='utf8').splitlines()]
def save(p,o):
 with p.open('x',encoding='utf8') as f:json.dump(o,f,ensure_ascii=False,indent=2)
def prepare():
 sources={'bridge':R.parent/'request_bridge_20260927/results/calls.jsonl','expansion':R.parent/'measurement_expansion_20260930/results/calls.jsonl'}
 bridge=readlines(sources['bridge']); expansion=readlines(sources['expansion']); pairs={}
 for r in bridge:
  pid=r['pair_id'];p=pairs.setdefault(pid,{'pair_id':pid,'question_id':r['question_id'],'phase':r['sample_phase'],'intent':{},'help':{}})
  if r['suite']=='intent':p['intent'][r['category']+'/'+r['arm']]={'status':r['status'],'value':r.get('parsed',{}).get('decision')}
 for r in expansion:
  if r['suite']=='helpfulness' and r['arm'] in ['joint_control','joint_no_examples']:
   pairs[r['pair_id']]['help'][r['arm']]={'status':r['status'],'value':r.get('parsed',{}).get('overall'),'evidence_ids':r.get('parsed',{}).get('evidence_ids',[])}
 save(R/'input.json',list(pairs.values()))
 save(R/'plan.json',{'question':'Does the observed association between expressed intent and first-answer helpfulness persist under alternative measurement pipelines?', 'design':'Seven nonexclusive intent indicators; direct vs question-first intent x original-example vs no-example helpfulness. Within each category and phase all four estimates use the same eligible pairs.', 'outcome':'substantial versus partial/unhelpful; uncertain, invalid and missing excluded explicitly, never recoded zero.', 'sample':'Previously selected answered development questions only; primary and extension reported separately, pooled descriptive only. No reserved80.', 'inference':'1000 paired question-level bootstrap resamples, seed314. Intervals conditional on observed sample and model labels, not annotation uncertainty. No causal or population claims; no multiple-testing significance claims.', 'rare_groups':'Below10 yes or10 no report counts but suppress difference/interval.', 'limitations':['Shared LLM measurement bias','Intent labels may overlap','Sample selection and restriction to answered questions','No verified answer correctness','No confounder adjustment; time/topic/author differences may explain associations'], 'next':'Use effect stability to prioritize label review and freeze measurement, not to select the most favorable effect.', 'source_sha256':{k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in sources.items()}})
 print('Prepared',len(pairs),'pairs')
def diff(rows,j):
 yes=[r[2+j%2] for r in rows if r[j//2]];no=[r[2+j%2] for r in rows if not r[j//2]]
 return sum(yes)/len(yes)-sum(no)/len(no) if yes and no else None
def run():
 start=time.time();data=json.loads((R/'input.json').read_text(encoding='utf8'));out=R/'results';out.mkdir(exist_ok=True)
 cats=sorted({k.split('/')[0] for p in data for k in p['intent']});rng=random.Random(314);results=[]
 for phase in ['primary','extension','pooled']:
  candidates=[p for p in data if phase=='pooled' or p['phase']==phase]
  for cat in cats:
   rows=[];excluded=collections.Counter();ids=[]
   for p in candidates:
    ins=[p['intent'].get(cat+'/'+a,{}) for a in ['direct','question_first']];hs=[p['help'].get(a,{}) for a in ['joint_control','joint_no_examples']]
    if any(x.get('status')!='valid' or x.get('value') not in ['yes','no'] for x in ins):excluded['intent_missing_invalid_uncertain']+=1;continue
    if any(x.get('status')!='valid' or x.get('value') not in ['substantial','partial','unhelpful'] for x in hs):excluded['help_missing_invalid_uncertain']+=1;continue
    rows.append(tuple([int(x['value']=='yes') for x in ins]+[int(x['value']=='substantial') for x in hs]));ids.append(p['pair_id'])
   item={'phase':phase,'category':cat,'candidate_n':len(candidates),'eligible_n':len(rows),'excluded':dict(excluded),'pair_ids':ids,'pipelines':[]}
   boot=[[] for _ in range(4)]
   eligible=[min(sum(r[j//2] for r in rows),sum(not r[j//2] for r in rows))>=10 for j in range(4)]
   if any(eligible):
    for b in range(1000):
     sample=rng.choices(rows,k=len(rows))
     for j in range(4):
      if eligible[j]:
       d=diff(sample,j)
       if d is not None:boot[j].append(d)
   for j in range(4):
    ys=[r[2+j%2] for r in rows if r[j//2]];ns=[r[2+j%2] for r in rows if not r[j//2]];bs=sorted(boot[j])
    item['pipelines'].append({'intent':['direct','question_first'][j//2],'helpfulness':['joint_control','joint_no_examples'][j%2],'yes_n':len(ys),'no_n':len(ns),'yes_substantial':sum(ys),'no_substantial':sum(ns),'risk_difference':diff(rows,j) if eligible[j] else None,'conditional_bootstrap95':[bs[int(.025*(len(bs)-1))],bs[int(.975*(len(bs)-1))]] if bs else None,'suppression':None if eligible[j] else 'one intent group below10'})
   estimates=[x['risk_difference'] for x in item['pipelines'] if x['risk_difference'] is not None];item['direction_changes_across_pipelines']=bool(estimates and min(estimates)<0<max(estimates))
   results.append(item)
   (out/'progress.json').write_text(json.dumps({'finished_cells':len(results),'planned_cells':3*len(cats)}))
 counts={a:collections.Counter(p['help'].get(a,{}).get('value','missing') for p in data) for a in ['joint_control','joint_no_examples']}
 evidence={a:sum(p['help'].get(a,{}).get('value') in ['partial','substantial'] and not p['help'].get(a,{}).get('evidence_ids') for p in data) for a in counts}
 save(out/'analysis.json',{'sample_pairs':len(data),'cells':results,'helpfulness_counts':counts,'positive_with_no_evidence_ids':evidence,'runtime_seconds':time.time()-start,'interpretation':'Exploratory associations conditional on selected development cases and LLM labels; not causal effects or label validation.'})
 print(json.dumps({'completed':len(results),'seconds':time.time()-start,'direction_change_cells':[(r['phase'],r['category']) for r in results if r['direction_changes_across_pipelines']]}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');a=p.parse_args();prepare() if a.prepare else run()
