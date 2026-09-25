"""Extract all real-case prompt disagreements; no new labels or held-out data."""
import collections
import json
from pathlib import Path

import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--root',type=Path,required=True)
parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args()
ROOT=args.root
OUT=args.out
OUT.mkdir(parents=True,exist_ok=True)
cases={r['case_id']:r for r in json.loads((ROOT/'data/cases.json').read_text(encoding='utf8')) if r['source']!='synthetic_diagnostic'}
summaries={}
for size in ['7b','14b']:
    arms=collections.defaultdict(dict)
    for line in (ROOT/'results_ssd'/size/'intent/calls.jsonl') .open(encoding='utf8'):
        r=json.loads(line)
        if r['case_id'] in cases and r['arm'] in ['full-shared_json-greedy','full-isolated_json-greedy']:
            arms[r['arm']][(r['case_id'],r['category'])]=r
    counts=collections.Counter()
    with (OUT/f'{size}_prompt_disagreements.jsonl').open('x') as f:
        a,b=arms['full-shared_json-greedy'],arms['full-isolated_json-greedy']
        for key in sorted(a.keys()&b.keys()):
            x,y=a[key],b[key]
            if x['status']!='valid' or y['status']!='valid':kind='invalid_output'
            elif x['parsed']['decision']!=y['parsed']['decision']:kind='decision_disagreement'
            else:continue
            counts[key[1]+'|'+kind]+=1
            f.write(json.dumps({'case_id':key[0],'category':key[1],'kind':kind,'question':cases[key[0]],'shared':{k:x.get(k) for k in ['status','parsed','errors','raw']},'isolated':{k:y.get(k) for k in ['status','parsed','errors','raw']},'review_status':'not_reviewed','reference_label':None})+'\n')
    dimensions=collections.defaultdict(lambda:collections.defaultdict(collections.Counter))
    for line in (ROOT/'results_ssd'/size/'boundary/calls.jsonl') .open(encoding='utf8'):
        r=json.loads(line)
        if r['condition']=='original' and r['status']=='valid':
            for dim in ['relevance','actionability','sufficiency','overall']:
                dimensions[r['variant']][dim][str(r['parsed'][dim])]+=1
    summaries[size]={'disagreement_counts':dict(counts),'original_helpfulness_dimensions':dimensions}
(OUT/'error_audit_summary.json').write_text(json.dumps(summaries,indent=2))
print(json.dumps(summaries,indent=2))
