"""Paired development sensitivity and downstream feasibility, no model selection by outcome."""
import collections
import datetime
import hashlib
import itertools
import json
import random
from pathlib import Path

import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--root',type=Path,required=True)
parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args()
ROOT=args.root
OUT=args.out
OUT.mkdir(parents=True,exist_ok=True)
CATEGORIES = ['api_usage','discrepancy','errors','review','conceptual','api_change','learning']

def bootstrap(values, seed=314):
    if not values:
        return None
    rng = random.Random(seed)
    stats = sorted(sum(rng.choices(values, k=len(values)))/len(values) for _ in range(2000))
    return {'mean': sum(values)/len(values), 'ci95_percentile': [stats[49], stats[1949]], 'n_questions': len(values)}

def paired(a, b):
    common = sorted(a.keys() & b.keys())
    usable = [q for q in common if all(a[q][c]['status']=='valid' and b[q][c]['status']=='valid' and a[q][c]['parsed']['decision'] in ['yes','no'] and b[q][c]['parsed']['decision'] in ['yes','no'] for c in CATEGORIES)]
    per = {}
    for c in CATEGORIES:
        counts = collections.Counter((a[q][c]['parsed']['decision'], b[q][c]['parsed']['decision']) for q in usable)
        yy,yn,ny,nn = (counts[x] for x in [('yes','yes'),('yes','no'),('no','yes'),('no','no')])
        per[c] = {'yes_yes':yy,'yes_no':yn,'no_yes':ny,'no_no':nn,'agreement': (yy+nn)/len(usable) if usable else None, 'positive_agreement':2*yy/(2*yy+yn+ny) if 2*yy+yn+ny else None}
    jaccard=[];delta=[]; exact=0; empty=0
    for q in usable:
        aa={c for c in CATEGORIES if a[q][c]['parsed']['decision']=='yes'}
        bb={c for c in CATEGORIES if b[q][c]['parsed']['decision']=='yes'}
        exact += aa==bb
        delta.append(len(bb)-len(aa))
        if aa|bb: jaccard.append(len(aa&bb)/len(aa|bb))
        else: empty+=1
    return {'joint_observed_questions':len(common),'joint_valid_binary_questions':len(usable),'excluded_invalid_uncertain_or_missing':len(common)-len(usable),'mean_yes_a':sum(sum(a[q][c]['parsed']['decision']=='yes' for c in CATEGORIES) for q in usable)/len(usable) if usable else None,'mean_yes_b':sum(sum(b[q][c]['parsed']['decision']=='yes' for c in CATEGORIES) for q in usable)/len(usable) if usable else None,'delta_b_minus_a_bootstrap':bootstrap(delta),'exact_label_set_agreement':exact/len(usable) if usable else None,'mean_jaccard_nonempty_union':sum(jaccard)/len(jaccard) if jaccard else None,'both_empty_sets':empty,'per_category':per}

def main():
    OUT.mkdir(exist_ok=True)
    dest=OUT/'results.json'
    if dest.exists(): raise RuntimeError('Preserve prior analysis; choose a new output version')
    result={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'protocol':'Complete primary arms only; paired real questions; exclude invalid/uncertain explicitly; question-level bootstrap 2000 draws seed314. No accuracy or causal claims. No held-out cases used. Shared vs isolated is a prompt-package comparison, not an isolated wording effect.','input_sha256':{},'models':{}}
    for size in ['7b','14b']:
        path=ROOT/'results_ssd'/size/'intent/calls.jsonl'
        result['input_sha256'][str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        groups=collections.defaultdict(lambda:collections.defaultdict(dict)); total=collections.Counter(); duplicates=0
        for line in path .open(encoding='utf8'):
            r=json.loads(line);total[r['status']]+=1
            if r['source']=='synthetic_diagnostic':continue
            q=r['case_id'];cat=r['category'];arm=r['arm']
            duplicates += cat in groups[arm][q]
            groups[arm][q][cat]={'status':r['status'],'parsed':r.get('parsed')}
        # Missing category records must not be silently treated as negatives.
        for group in groups.values():
            for row in group.values():
                for c in CATEGORIES:row.setdefault(c,{'status':'missing','parsed':None})
        comparisons={}
        for seed in ['greedy','sample314','sample2718']:
            for view in ['full','prose','title']:
                a=f'{view}-shared_json-{seed}';b=f'{view}-isolated_json-{seed}'
                if a in groups and b in groups:comparisons[a+'__'+b]=paired(groups[a],groups[b])
            for prompt in ['shared_json','isolated_json']:
                for view in ['prose','title']:
                    a=f'full-{prompt}-{seed}';b=f'{view}-{prompt}-{seed}'
                    if a in groups and b in groups:comparisons[a+'__'+b]=paired(groups[a],groups[b])
        for view in ['full','prose','title']:
            for prompt in ['shared_json','isolated_json']:
                for seed in ['sample314','sample2718']:
                    a=f'{view}-{prompt}-greedy';b=f'{view}-{prompt}-{seed}'
                    if a in groups and b in groups:comparisons[a+'__'+b]=paired(groups[a],groups[b])
        # Use only the boundary suite once: originals repeat in the control suite.
        hp=ROOT/'results_ssd'/size/'boundary/calls.jsonl'
        result['input_sha256'][str(hp)]=hashlib.sha256(hp.read_bytes()).hexdigest()
        helpful=collections.defaultdict(dict)
        for line in hp .open(encoding='utf8'):
            r=json.loads(line)
            if r['condition']=='original':helpful[r['variant']][r['pair_id']]=r
        bridge={}
        for variant,hrows in helpful.items():
            for prompt in ['shared_json','isolated_json']:
                arm=f'full-{prompt}-greedy';g=groups[arm]
                common=sorted(hrows.keys()&g.keys())
                valid=[q for q in common if hrows[q]['status']=='valid' and all(g[q][c]['status']=='valid' and g[q][c]['parsed']['decision'] in ['yes','no'] for c in CATEGORIES)]
                ratings=collections.Counter(hrows[q]['parsed']['overall'] for q in valid)
                tables={c:dict(collections.Counter(g[q][c]['parsed']['decision']+'|'+hrows[q]['parsed']['overall'] for q in valid)) for c in CATEGORIES}
                bridge[variant+'__'+arm]={'matched_original_questions':len(common),'valid_binary_complete_questions':len(valid),'overall_distribution':dict(ratings),'contingency_counts':tables,'inference':'Feasibility only: 20 development pairs maximum, sparse cells and potential ceiling. No significance tests or regression fit.'}
        result['models'][size]={'all_call_statuses':dict(total),'duplicate_keys':duplicates,'paired_comparisons':comparisons,'downstream_feasibility':bridge}
    dest.write_text(json.dumps(result,indent=2))
    for size,m in result['models'].items():
        print(size,m['all_call_statuses'])
        for key in ['full-shared_json-greedy__full-isolated_json-greedy','full-shared_json-greedy__prose-shared_json-greedy','full-isolated_json-greedy__title-isolated_json-greedy','full-isolated_json-greedy__full-isolated_json-sample314']:
            r=m['paired_comparisons'][key]
            print(key,json.dumps({k:v for k,v in r.items() if k!='per_category'}))
        print('BRIDGE',json.dumps({k:{x:y for x,y in v.items() if x!='contingency_counts'} for k,v in m['downstream_feasibility'].items()}))

if __name__=='__main__':main()
