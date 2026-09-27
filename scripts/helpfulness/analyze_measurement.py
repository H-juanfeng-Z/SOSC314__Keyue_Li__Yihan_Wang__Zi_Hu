"""CPU-only development diagnostics; not ground-truth accuracy."""
import argparse
import collections
import json
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--metadata',type=Path,required=True)
    a=p.parse_args()
    rows=[json.loads(s) for s in (a.run/'calls.jsonl').read_text(encoding='utf8').splitlines()]
    index={(r['suite'],r['id'],r['arm']):r for r in rows}
    assert len(index)==len(rows)
    audit={r['audit_id']:r for r in json.loads((a.metadata/'intent_selection.json').read_text(encoding='utf8'))}
    meta={r['pair_id']:r for r in json.loads((a.metadata/'sampling_metadata.json').read_text(encoding='utf8'))}
    out={'calls':len(rows),'statuses':dict(collections.Counter(r['suite']+'/'+r['status'] for r in rows)),
         'interpretation':'Same-model blinded reannotation and stratified development sensitivity; not accuracy or population prevalence.',
         'helpfulness_cells':{},'helpfulness_paired':{},'intent_review':{},'oversized':[]}
    for r in rows:
        if r['status']=='context_skipped':out['oversized'].append({k:r[k] for k in ['suite','id','arm','input_tokens']})
    for arm in ['baseline','evidence']:
        valid=[r for r in rows if r['suite']=='helpfulness_development' and r['arm']==arm and r['status']=='valid']
        strata=collections.defaultdict(collections.Counter)
        for r in valid:
            m=meta[r['id']]
            for field in ['year_band','code_band','score_band']:
                strata[field+'/'+m[field]][r['parsed']['overall']]+=1
        out['helpfulness_cells'][arm]={'valid':len(valid),'overall':dict(collections.Counter(r['parsed']['overall'] for r in valid)),
            'dimensions_equal':sum(r['parsed']['actionability']==r['parsed']['sufficiency'] for r in valid),
            'stratified_counts':dict(strata)}
    transitions=collections.Counter();excluded=collections.Counter()
    for pid in meta:
        x=index.get(('helpfulness_development',pid,'baseline'));y=index.get(('helpfulness_development',pid,'evidence'))
        if not x or not y:excluded['missing']+=1;continue
        if x['status']!='valid' or y['status']!='valid':excluded['invalid_or_skipped']+=1;continue
        transitions[x['parsed']['overall']+' -> '+y['parsed']['overall']]+=1
    out['helpfulness_paired']={'transitions':transitions,'excluded':excluded}
    cells=collections.defaultdict(collections.Counter)
    for aid,selection in audit.items():
        category=aid.split('__')[-1]
        x=index.get(('intent_audit',aid,'evidence_first'));y=index.get(('intent_audit',aid,'boundary_check'))
        cell=cells[category+'/'+selection['selection_group']]
        if not x or not y:cell['missing']+=1;continue
        if x['status']!='valid' or y['status']!='valid':cell['invalid_or_skipped']+=1;continue
        cell[x['parsed']['decision']+' -> '+y['parsed']['decision']]+=1
    out['intent_review']=dict(cells)
    with (a.run/'analysis.json').open('x',encoding='utf8') as f:json.dump(out,f,indent=2)
    print(json.dumps(out,indent=2))

if __name__=='__main__':main()
