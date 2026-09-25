"""Recompute descriptive cells from text-free, per-call decision exports."""
import argparse
import collections
import hashlib
import json
from pathlib import Path

def summarize(path):
    rows = [json.loads(line) for line in path.read_text(encoding='utf8').splitlines()]
    keys = [(r['pair_id'], r['condition'], r.get('arm', r.get('variant'))) for r in rows]
    assert len(keys) == len(set(keys)), 'Duplicate decisions'
    cells = {}
    for arm, condition in sorted({(r.get('arm', r.get('variant')), r['condition']) for r in rows}):
        rr = [r for r in rows if r.get('arm', r.get('variant')) == arm and r['condition'] == condition]
        valid = [r for r in rr if r['status'] == 'valid']
        numeric = [r for r in valid if all(type(r['parsed'][d]) is int for d in ['actionability','sufficiency'])]
        cells[arm+'/'+condition] = {
            'calls':len(rr), 'valid':len(valid),
            'overall':dict(collections.Counter(r['parsed']['overall'] for r in valid)),
            'numeric_dimension_pairs':len(numeric),
            'dimensions_different':sum(r['parsed']['actionability'] != r['parsed']['sufficiency'] for r in numeric),
            'marker_citations':sum(r.get('cites_marker_only_line',False) for r in valid)}
    return {'calls':len(rows), 'statuses':dict(collections.Counter(r['status'] for r in rows)),
            'export_sha256':hashlib.sha256(path.read_bytes()).hexdigest(), 'cells':cells}

def main():
    repo = Path(__file__).resolve().parents[2]
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=repo/'tables/annotation_diagnostics/decisions')
    p.add_argument('--out',type=Path,default=repo/'tables/annotation_diagnostics/helpfulness_summary.json')
    a=p.parse_args()
    result={'interpretation':'Development sensitivity, not annotation accuracy. Invalid outputs excluded from rating counts and reported separately. Conditions have different eligible samples.'}
    for suite in ['content_sensitivity','code_grounding','rubric_factorial']:
        result[suite]=summarize(a.root/(suite+'.jsonl'))
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(a.out)

if __name__=='__main__': main()
