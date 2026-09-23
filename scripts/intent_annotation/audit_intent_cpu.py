"""CPU-only audit; never alter labels or manufacture human ground truth."""
import argparse
import collections
import json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
root=a.root; results={}; indexes={}
constraints=json.loads((root/'intent_round3/data/diagnostic_constraints.json').read_text())
for rnd in ('intent_round3','intent_round4'):
    for folder in sorted((root/rnd/'results').glob('*')):
        file=folder/'calls.jsonl'
        if not file.exists(): continue
        rows=[]; malformed=0
        for line in file.read_text(encoding='utf8').splitlines():
            try: rows.append(json.loads(line))
            except ValueError: malformed+=1
        key=rnd+'/'+folder.name
        idx={(r['case_id'],r['category']):r for r in rows}; indexes[key]=idx
        stats={'calls':len(rows),'expected':1036,'unique_calls':len(idx),'complete':(folder/'summary.json').exists(),'malformed_lines':malformed,'strata':{}}
        for source in {r['source'] for r in rows}:
            rr=[r for r in rows if r['source']==source]
            stats['strata'][source]={'status':dict(collections.Counter(r['status'] for r in rr)),
              'errors':dict(collections.Counter(e for r in rr for e in r.get('errors',[]))),
              'decision_only':dict(collections.Counter(r.get('parsed',{}).get('decision','unparsed') for r in rr))}
        checks=[]
        for c in constraints:
            for dec,k in [('yes','expected_yes'),('no','expected_no')]:
                for cat in c[k]:
                    r=idx.get((c['case_id'],cat),{}); observed=r.get('parsed',{}).get('decision')
                    checks.append({'case_id':c['case_id'],'category':cat,'expected':dec,'observed':observed,'strict':r.get('status')=='valid' and observed==dec,'decision_only':observed==dec})
        stats['diagnostic_checks']=checks; results[key]=stats
out=root/'intent_round4/audit';out.mkdir(exist_ok=True)
real=json.loads((root/'intent_round3/data/cases.json').read_text())
# Deterministic IDs, selected for model disagreement; not a representative accuracy sample.
disagreement=[]
for c in real:
    if c['source']=='synthetic_diagnostic':continue
    for cat in ['api_usage','discrepancy','errors','review','conceptual','api_change','learning']:
        values={k:v.get((c['case_id'],cat),{}).get('parsed',{}).get('decision') for k,v in indexes.items()}
        if len(set(values.values())-{None})>1:disagreement.append({'case_id':c['case_id'],'category':cat,'model_decisions':values})
selected=sorted({d['case_id'] for d in disagreement})[:40]
review=[{**c,'human_labels':None,'human_evidence':None,'notes':''} for c in real if c['case_id'] in selected]
(out/'comparison.json').write_text(json.dumps(results,indent=2),encoding='utf8')
(out/'disagreements.json').write_text(json.dumps(disagreement,indent=2),encoding='utf8')
# Do not overwrite a reviewer-filled file on a later audit.
reviewfile=out/'review_candidates_blank.json'
if not reviewfile.exists(): reviewfile.write_text(json.dumps(review,indent=2),encoding='utf8')
print(json.dumps({k:{'calls':v['calls'],'complete':v['complete'],'strict_constraints':sum(t['strict'] for t in v['diagnostic_checks']),'decision_only_constraints':sum(t['decision_only'] for t in v['diagnostic_checks'])} for k,v in results.items()},indent=2))
