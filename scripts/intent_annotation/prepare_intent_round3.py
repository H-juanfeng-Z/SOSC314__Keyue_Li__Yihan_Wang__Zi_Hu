"""Deterministic development expansion and AI-authored diagnostic controls."""
import hashlib
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser()
parser.add_argument('--pairs', type=Path, default=ROOT/'data/interim/helpfulness/blinded_pairs.json')
parser.add_argument('--model-sample', type=Path, required=True)
parser.add_argument('--out', type=Path, default=ROOT/'data/interim/intent_round3')
args = parser.parse_args()
OUT = args.out
OUT.mkdir(parents=True, exist_ok=True)
old = json.loads(args.pairs.read_text(encoding='utf-8'))
excluded = {c['question_id'] for c in old}
cases = [{'case_id':c['pair_id'],'question_id':c['question_id'],'source':'original_development',
          'title':c['title'],'text':c['question_text']} for c in old[:20]]
path = args.model_sample
pool = []
with path.open(encoding='utf-8') as f:
    for line in f:
        c = json.loads(line)
        if c['question_id'] not in excluded:
            pool.append({k:c[k] for k in ('question_id','title','natural_language_text','code_text')})
pool.sort(key=lambda c:hashlib.sha256(('intent-r3-20260920:'+str(c['question_id'])).encode()).hexdigest())
for c in pool[:100]:
    cases.append({'case_id':'NEW-'+str(c['question_id']),'question_id':c['question_id'],
        'source':'expanded_development','title':c['title'],
        'text':c['natural_language_text']+'\n[EXTRACTED CODE; ORIGINAL POSITION NOT PRESERVED]\n'+c['code_text']})
contexts = ['processing CSV rows with a Python library','drawing a chart with a Python library',
            'sending an HTTP request with a Python library','traversing a graph with a Python library']
requests = {
 'api_usage':'I have not implemented this task yet. Please show the concrete steps and an example that accomplishes it.',
 'discrepancy':'My implementation runs without an exception, but returns an empty result instead of the expected three items. Please find why the output is wrong and fix it.',
 'errors':'My implementation raises TypeError: invalid argument type. Please diagnose and fix this exception.',
 'review':'My implementation already gives the correct results. Please compare possible implementations and recommend a more efficient approach.',
 'conceptual':'I do not need an implementation or bug fix. Please explain the underlying design principle and why this mechanism works.',
 'api_change':'The same call worked in version 1 of the library, but fails after upgrading to version 2. Please explain the API change and how to migrate.',
 'learning':'Please recommend tutorials or documentation so I can study this topic. I am not requesting a solution to a specific implementation.'}
controls = []
for topic, context in enumerate(contexts):
    for label, request in requests.items():
        cid=f'SYN-{topic+1}-{label}'
        cases.append({'case_id':cid,'question_id':None,'source':'synthetic_diagnostic',
                      'title':'Question about '+context,'text':'Context: '+context+'.\n'+request})
        # Only explicit positive and justified negative constraints, not a complete gold taxonomy.
        negatives = ['learning'] if label != 'learning' else []
        controls.append({'case_id':cid,'expected_yes':[label],'expected_no':negatives,
                         'author':'AI-authored diagnostic; not independently human validated'})
(OUT/'cases.json').write_text(json.dumps(cases,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'diagnostic_constraints.json').write_text(json.dumps(controls,indent=2),encoding='utf-8')
manifest = {'real_questions':120,'synthetic_questions':28,'excluded_question_ids':sorted(excluded),
    'source_sample_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
    'cases_sha256':hashlib.sha256((OUT/'cases.json').read_bytes()).hexdigest(),
    'selection':'First 100 SHA256-ranked IDs excluding all original 100 pairs; no outcome-based selection',
    'representation':'Original 20 retain readable code positions; new 100 use existing prose plus separately extracted code. Analyze source groups separately.',
    'human_gold_labels':False,'held_out_80_used':False}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
assert len({c['question_id'] for c in cases if c['source']!='synthetic_diagnostic'})==120
assert not any(c['question_id'] in excluded for c in cases if c['source']=='expanded_development')
print(json.dumps({k:v for k,v in manifest.items() if k!='excluded_question_ids'},indent=2))
