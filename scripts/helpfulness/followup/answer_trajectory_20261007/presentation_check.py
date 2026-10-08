"""Descriptive within-intent feature associations, not adjusted causal effects."""
import json,statistics
from pathlib import Path
R=Path(__file__).resolve().parent
features={p['pair_id']:p for p in json.loads((R/'question_features.json').read_text(encoding='utf8'))}
data=json.loads((R.parent/'association_sensitivity_20261007/input.json').read_text(encoding='utf8'))
cut=statistics.median(p['readable_whitespace_tokens'] for p in features.values());out=[]
cats=sorted({k.split('/')[0] for p in data for k in p['intent']})
for cat in cats:
 for ha in ['joint_control','joint_no_examples']:
  pool=[p for p in data if p['intent'].get(cat+'/direct',{}).get('value')=='yes' and p['intent'][cat+'/direct']['status']=='valid' and p['help'].get(ha,{}).get('status')=='valid' and p['help'][ha]['value'] in ['substantial','partial','unhelpful']]
  for feature in ['code_present','link_present','longer_than_median']:
   groups={True:[],False:[]}
   for p in pool:
    f=features[p['pair_id']];flag={'code_present':f['code_block_count']>0,'link_present':f['link_count']>0,'longer_than_median':f['readable_whitespace_tokens']>cut}[feature];groups[flag].append(p['help'][ha]['value']=='substantial')
   a,b=groups[True],groups[False];out.append({'intent':cat,'helpfulness':ha,'feature':feature,'present_n':len(a),'absent_n':len(b),'present_substantial':sum(a),'absent_substantial':sum(b),'difference':sum(a)/len(a)-sum(b)/len(b) if min(len(a),len(b))>=10 else None})
(R/'results').mkdir(exist_ok=True)
with (R/'results/presentation_analysis.json').open('x',encoding='utf8') as f:json.dump({'median_readable_tokens':cut,'comparisons':out,'limits':'Descriptive pooled development data; overlapping intent strata; no year/topic adjustment; edited snapshots may postdate answers; whitespace token count not linguistic word count; no significance claims.'},f,indent=2)
print('Completed',len(out),'descriptive comparisons')
