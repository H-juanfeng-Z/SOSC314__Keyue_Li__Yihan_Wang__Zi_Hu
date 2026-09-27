"""Paired measurement sensitivity and intent/helpfulness cross-tabs, no causal inference."""
import argparse
import collections
import json
from pathlib import Path

def summarize(root):
    rows=[json.loads(x) for x in (root/'calls.jsonl').read_text(encoding='utf8').splitlines()]
    idx={(r['pair_id'],r['suite'],r['arm'],r.get('category')):r for r in rows}
    assert len(idx)==len(rows),'duplicate calls'
    pids=sorted({r['pair_id'] for r in rows})
    categories=sorted({r['category'] for r in rows if r['suite']=='intent'})
    result={'records':len(rows),'inference_calls':sum(r.get('inference_requested', 'messages' in r) for r in rows),
      'generation_calls':sum('output_tokens' in r for r in rows),'statuses':dict(collections.Counter(r['status'] for r in rows)),
      'input_tokens':sum(r.get('input_tokens',0) for r in rows),'output_tokens':sum(r.get('output_tokens',0) for r in rows),
      'interpretation':'Measurement association feasibility only; no human accuracy, no causal inference, shared-model bias possible.',
      'samples':{}}
    for phase in ['primary','extension']:
        ids={r['pair_id'] for r in rows if r['sample_phase']==phase}
        out={'observed_pairs':len(ids),'intent_transitions':collections.defaultdict(collections.Counter),
             'helpfulness_transitions':collections.Counter(),'missing_or_invalid_pairs':collections.Counter(),
             'cross_tabs':collections.defaultdict(collections.Counter),'dimension_counts':collections.defaultdict(collections.Counter)}
        for pid in sorted(ids):
            for cat in categories:
                x=idx.get((pid,'intent','direct',cat));y=idx.get((pid,'intent','question_first',cat))
                if x and y and x['status']==y['status']=='valid':out['intent_transitions'][cat][x['parsed']['decision']+' -> '+y['parsed']['decision']]+=1
                else:out['missing_or_invalid_pairs']['intent/'+cat]+=1
            x=idx.get((pid,'helpfulness','direct',None));y=idx.get((pid,'helpfulness','question_first',None))
            if x and y and x['status']==y['status']=='valid':out['helpfulness_transitions'][x['parsed']['overall']+' -> '+y['parsed']['overall']]+=1
            else:out['missing_or_invalid_pairs']['helpfulness']+=1
            for arm in ['direct','question_first']:
                h=idx.get((pid,'helpfulness',arm,None))
                if not h or h['status']!='valid':continue
                out['dimension_counts'][arm]['valid']+=1
                if all(type(h['parsed'][d]) is int for d in ['actionability','sufficiency']):
                    out['dimension_counts'][arm]['numeric_pairs']+=1
                    out['dimension_counts'][arm]['equal']+=h['parsed']['actionability']==h['parsed']['sufficiency']
                for cat in categories:
                    r=idx.get((pid,'intent',arm,cat))
                    if r and r['status']=='valid':out['cross_tabs'][arm+'/'+cat][r['parsed']['decision']+'|'+h['parsed']['overall']]+=1
        result['samples'][phase]=out
    (root/'analysis.json').write_text(json.dumps(result,indent=2))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    print(json.dumps(summarize(p.parse_args().root),indent=2))
