"""Rebuild figures directly from complete raw logs; no model inference or relabeling."""
from pathlib import Path
import json
import hashlib
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=Path, required=True)
ROOT = parser.parse_args().root
OUT=ROOT/'report_outputs';OUT.mkdir(exist_ok=True)
constraints=json.loads((ROOT/'intent_round3/diagnostic_constraints.json').read_text())
categories=['api_usage','discrepancy','errors','review','conceptual','api_change','learning']
names=['Implementation','Unexpected behavior','Explicit error','Review / comparison','Concept explanation','API / version change','Learning resources']
labels=[];metrics=[];heat=[];missing=[];hashes={}
for size in ['7','3']:
    for variant,label in [('examples','Baseline'),('shared_json','+ JSON examples'),('isolated','Single-target'),('isolated_json','Single-target + JSON')]:
        rnd=3 if variant=='examples' else 4
        folder=ROOT/f'intent_round{rnd}_downloaded/results/{size}b-{variant}'
        path=folder/'calls.jsonl';rows=[json.loads(l) for l in path.read_text(encoding='utf8').splitlines()]
        assert (folder/'summary.json').exists() and len(rows)==1036
        idx={(r['case_id'],r['category']):r for r in rows};assert len(idx)==1036
        hashes[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
        checks=[]
        for c in constraints:
            for decision,key in [('yes','expected_yes'),('no','expected_no')]:
                for cat in c[key]:
                    r=idx[c['case_id'],cat];match=r.get('parsed',{}).get('decision')==decision
                    checks.append((match,match and r['status']=='valid'))
        real=[r for r in rows if r['source']!='synthetic_diagnostic'];assert len(real)==840
        counts=[sum(r['category']==cat and r['status']=='valid' and r['parsed']['decision']=='yes' for r in real) for cat in categories]
        invalid=sum(r['status']!='valid' for r in real)
        m={'model':size+'B','variant':variant,'valid':sum(r['status']=='valid' for r in rows),'calls':len(rows),
           'strict_constraints':sum(t[1] for t in checks),'decision_constraints':sum(t[0] for t in checks),'constraints':len(checks),
           'real_invalid':invalid,'real_calls':840,'valid_yes_counts':dict(zip(categories,counts))}
        metrics.append(m);labels.append(size+'B  '+label);heat.append(counts);missing.append(invalid)
(OUT/'figure_data.json').write_text(json.dumps({'metrics':metrics,'source_sha256':hashes},indent=2),encoding='utf8')
