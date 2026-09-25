"""Paired content sensitivity summaries; excludes unchanged/empty variants."""
import argparse
import collections
import json
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
manifest=json.loads((a.root/'transformation_manifest.json').read_text(encoding='utf8'))
flags={(c['pair_id'],v['condition']):v for c in manifest['cases'] for v in c['variants']}
rows=[json.loads(line) for line in (a.root/'calls.jsonl').read_text(encoding='utf8').splitlines()]
idx={(r['pair_id'],r['condition'],r['variant']):r for r in rows}
assert len(idx)==len(rows),'Duplicate decisions'
result={'calls':len(rows),'complete':len(rows)==160,'status_counts':dict(collections.Counter(r['status'] for r in rows)),'comparisons':{},'note':'Not accuracy or causal effect. Removal has no predetermined quality order. Subsets differ by transformation; invalid, null and uncertain excluded explicitly.'}
for prompt in ['boundary','boundary_json']:
    for condition in ['prose_only','code_only','prefix_half']:
        summary={'n_original_cases':20,'excluded':collections.Counter(),'overall_transitions':collections.Counter(),'dimension_changes':{d:[] for d in ['relevance','actionability','sufficiency']}}
        for case in manifest['cases']:
            q=case['pair_id'];flag=flags[q,condition]
            if flag['unchanged']:summary['excluded']['unchanged']+=1;continue
            if flag['empty']:summary['excluded']['empty']+=1;continue
            x=idx.get((q,'original',prompt));y=idx.get((q,condition,prompt))
            if x is None or y is None:summary['excluded']['missing']+=1;continue
            if x['status']!='valid' or y['status']!='valid':summary['excluded']['invalid_output']+=1;continue
            xx,yy=x['parsed'],y['parsed']
            summary['overall_transitions'][xx['overall']+' -> '+yy['overall']]+=1
            for dim in summary['dimension_changes']:
                if xx[dim] is not None and yy[dim] is not None:summary['dimension_changes'][dim].append(yy[dim]-xx[dim])
        summary['dimension_changes']={d:{'n':len(v),'lower':sum(x<0 for x in v),'same':sum(x==0 for x in v),'higher':sum(x>0 for x in v),'mean_change':sum(v)/len(v) if v else None} for d,v in summary['dimension_changes'].items()}
        result['comparisons'][prompt+'__'+condition]=summary
out=a.root/'paired_content_analysis.json'
if out.exists():raise RuntimeError('Refuse overwrite')
out.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
