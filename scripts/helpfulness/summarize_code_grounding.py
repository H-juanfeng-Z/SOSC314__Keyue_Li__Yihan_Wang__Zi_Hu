"""Paired code substitution diagnostics; no ground-truth quality assumption."""
import collections
import json
from pathlib import Path

import argparse
p=argparse.ArgumentParser()
p.add_argument('--root',type=Path,required=True)
root=p.parse_args().root
rows=[json.loads(x) for x in (root/'calls.jsonl').read_text(encoding='utf8').splitlines()]
idx={(r['pair_id'],r['condition'],r['variant']):r for r in rows}
assert len(idx)==len(rows)
out={'calls':len(rows),'statuses':dict(collections.Counter(r['status'] for r in rows)),'cells':{},'paired':{},'invalid':[]}
rank={'unhelpful':0,'partial':1,'substantial':2}
for r in rows:
    if r['status']!='valid':out['invalid'].append({k:r.get(k) for k in ['pair_id','condition','variant','errors']})
for prompt in ['boundary','boundary_json']:
    for condition in ['original','swapped_code','code_only','foreign_code_only']:
        rr=[r for r in rows if r['variant']==prompt and r['condition']==condition]
        valid=[r for r in rr if r['status']=='valid']
        out['cells'][prompt+'/'+condition]={'calls':len(rr),'valid':len(valid),'overall':dict(collections.Counter(r['parsed']['overall'] for r in valid)),'dimensions_equal':sum(r['parsed']['actionability']==r['parsed']['sufficiency'] for r in valid)}
    for a,b in [('original','swapped_code'),('code_only','foreign_code_only')]:
        pairs=[];trans=collections.Counter();high=[]
        for q in sorted({r['pair_id'] for r in rows}):
            x,y=idx[q,a,prompt],idx[q,b,prompt]
            if x['status']!='valid' or y['status']!='valid':continue
            xx,yy=x['parsed']['overall'],y['parsed']['overall'];trans[xx+' -> '+yy]+=1
            if xx in rank and yy in rank:pairs.append(rank[yy]-rank[xx])
            if yy=='substantial':high.append({'pair_id':q,'donor':y.get('donor_pair'),'reason':y['parsed']['reason']})
        out['paired'][prompt+'/'+a+'__'+b]={'joint_valid':sum(trans.values()),'rankable':len(pairs),'lower':sum(d<0 for d in pairs),'same':sum(d==0 for d in pairs),'higher':sum(d>0 for d in pairs),'transitions':dict(trans),'high_donor_cases':high}
out['interpretation']='Different-question donor code is not automatically ground-truth irrelevant. Paired comparisons use jointly valid outputs; no accuracy claim.'
(root/'analysis.json').write_text(json.dumps(out,indent=2),encoding='utf8')
print(json.dumps(out,indent=2))
