"""Outcome-threshold sensitivity on the same complete-case support; no new labels."""
import json,collections
from pathlib import Path
R=Path(__file__).resolve().parent;data=json.loads((R/'input.json').read_text(encoding='utf8'));rows=[]
cats=sorted({k.split('/')[0] for p in data for k in p['intent']})
for phase in ['primary','extension','pooled']:
 for cat in cats:
  sample=[]
  for p in data:
   if phase!='pooled' and p['phase']!=phase:continue
   ins=[p['intent'].get(cat+'/'+a,{}) for a in ['direct','question_first']];hs=[p['help'].get(a,{}) for a in ['joint_control','joint_no_examples']]
   if all(x.get('status')=='valid' and x.get('value') in ['yes','no'] for x in ins) and all(x.get('status')=='valid' and x.get('value') in ['partial','substantial','unhelpful'] for x in hs):sample.append(p)
  for ia in ['direct','question_first']:
   for ha in ['joint_control','joint_no_examples']:
    item={'phase':phase,'category':cat,'intent':ia,'help':ha,'n':len(sample),'thresholds':{}}
    for name,positive in [('any_help',['partial','substantial']),('substantial',['substantial'])]:
     groups={label:[int(p['help'][ha]['value'] in positive) for p in sample if p['intent'][cat+'/'+ia]['value']==label] for label in ['yes','no']}
     counts={k:{'n':len(v),'positive':sum(v),'negative':len(v)-sum(v)} for k,v in groups.items()}
     d=sum(groups['yes'])/len(groups['yes'])-sum(groups['no'])/len(groups['no']) if min(map(len,groups.values()),default=0)>=10 else None
     item['thresholds'][name]={'counts':counts,'risk_difference':d,'sparse_outcome_warning':any(min(sum(v),len(v)-sum(v))<5 for v in groups.values())}
    vals=[v['risk_difference'] for v in item['thresholds'].values()];item['direction_reverses']=all(v is not None for v in vals) and vals[0]*vals[1]<0;rows.append(item)
out={'comparisons':rows,'interpretation':'Descriptive threshold sensitivity on identical complete cases. Any-help negatives may be too rare for useful modeling. No significance tests, no accuracy or causal claims.'}
with (R/'results/threshold_analysis.json').open('x',encoding='utf8') as f:json.dump(out,f,indent=2)
print(json.dumps({'comparisons':len(rows),'direction_reversals':sum(x['direction_reverses'] for x in rows),'any_help_sparse':sum(x['thresholds']['any_help']['sparse_outcome_warning'] for x in rows)}))
