"""Recompute paired scope-calibration results from exported decisions."""
import argparse
import collections
import json
from pathlib import Path

def summarize(root):
    rows=[json.loads(s) for s in (root/'calls.jsonl').read_text(encoding='utf8').splitlines()]
    index={(r['suite'],r['id'],r['arm']):r for r in rows}
    assert len(index)==len(rows)
    result={'records':len(rows),'comparisons':{}}
    for suite,control,field in [('intent_audit','boundary_control','decision'),('helpfulness_development','evidence_control','overall')]:
        counts=collections.Counter();excluded=0
        for case in sorted({r['id'] for r in rows if r['suite']==suite}):
            a=index.get((suite,case,control));b=index.get((suite,case,'scope_revised'))
            if not a or not b or a['status']!='valid' or b['status']!='valid':
                excluded+=1;continue
            counts[a['parsed'][field]+' -> '+b['parsed'][field]]+=1
        result['comparisons'][suite]={'transitions':dict(counts),'excluded':excluded}
    (root/'analysis.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    print(json.dumps(summarize(parser.parse_args().root),indent=2))
