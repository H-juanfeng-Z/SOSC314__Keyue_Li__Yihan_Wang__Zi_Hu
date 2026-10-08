"""Compare frozen GPT evidence review with prior Qwen outputs. No accuracy claim."""
import json, hashlib
from collections import Counter
from pathlib import Path
R=Path(__file__).resolve().parent; B=R.parent
def read(p): return json.loads(p.read_text(encoding='utf8'))
def save(name,v): (R/name).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf8')
receipt=read(R/'freeze_receipt.json')
for name, digest in receipt['hashes'].items():
    assert hashlib.sha256((R/name).read_bytes()).hexdigest()==digest, name
reviews={r['case_id']:r for p in sorted(R.glob('review_0[1-5].json')) for r in read(p)}
packet={r['case_id']:r for r in read(R/'review_packet.json')}
mapping=read(R/'mapping_private.json')
models={}
paths={'14B':B/'answer_trajectory_20261007/results/calls.jsonl','7B':B/'integrated_20261008/answers7/results/calls.jsonl'}
for model,path in paths.items():
    rows=[json.loads(line) for line in path.read_text(encoding='utf8').splitlines() if line.strip()]
    assert len({r['pair_id'] for r in rows})==len(rows)
    models[model]={r['pair_id']:r for r in rows}
def pattern(labels):
    if any(x in (None,'uncertain') for x in labels): return 'indeterminate'
    a,b=[x=='substantial' for x in labels]
    return 'first_not_second_substantial' if not a and b else 'both_substantial' if a and b else 'first_substantial_second_not' if a else 'neither_substantial'
cases=[]
for m in mapping:
    rev=reviews[m['case_id']]; answers=[]
    for letter, meta in m['answers'].items():
        qwen={}
        for model,index in models.items():
            source=index[meta['pair_id']]
            qwen[model]={'status':source['status'],'label':source.get('parsed',{}).get('overall') if source.get('parsed') else None,'errors':source.get('errors',[])}
        answers.append(dict(letter=letter,metadata=meta,gpt=rev['answers'][letter],qwen=qwen))
    answers.sort(key=lambda x:x['metadata']['answer_rank'])
    assert [a['metadata']['answer_rank'] for a in answers]==[1,2]
    patterns={'GPT':pattern([a['gpt']['label'] for a in answers])}
    for model in models:
        patterns[model]=pattern([a['qwen'][model]['label'] if a['qwen'][model]['status']=='valid' else None for a in answers])
    assert (patterns['14B']=='first_not_second_substantial')==m['added']
    cases.append(dict(case_id=m['case_id'],question_id=m['question_id'],title=packet[m['case_id']]['title'],
        added=m['added'],disagreement=m['disagreement'],control=m['control'],flags=rev['flags'],request=rev['request'],answers=answers,patterns=patterns))
strata={}
for name,subset in [('all',cases),('added14',[c for c in cases if c['added']]),('disagreement14',[c for c in cases if c['disagreement']]),('target_union23',[c for c in cases if not c['control']]),('controls8',[c for c in cases if c['control']])]:
    strata[name]={'n':len(subset),'gpt_patterns':dict(Counter(c['patterns']['GPT'] for c in subset)),
        'case_ids_by_gpt_pattern':{p:[c['case_id'] for c in subset if c['patterns']['GPT']==p] for p in sorted({c['patterns']['GPT'] for c in subset})}}
agreement={}
for model in models:
    pairs=[(a['gpt']['label'],a['qwen'][model]['label']) for c in cases for a in c['answers'] if a['qwen'][model]['status']=='valid' and a['gpt']['label']!='uncertain' and a['qwen'][model]['label'] not in (None,'uncertain')]
    agreement[model]={'comparable_answers':len(pairs),'exact_label_agreement':sum(x==y for x,y in pairs),'substantial_binary_agreement':sum((x=='substantial')==(y=='substantial') for x,y in pairs),'cross_tab_gpt_to_qwen':dict(Counter(x+' -> '+y for x,y in pairs))}
summary={'cases':len(cases),'answers':sum(len(c['answers']) for c in cases),'gpt_labels':dict(Counter(a['gpt']['label'] for c in cases for a in c['answers'])),
    'strata':strata,'agreement_descriptive_not_accuracy':agreement,'flags':dict(Counter(f for c in cases for f in c['flags'])),
    'limitations':['Targeted sampling and eight non-target controls; not representative accuracy.','Same-context GPT review, not strict blinded independent validation or human gold.','Technical code not executed and external linked contents not inspected.','Archived edited questions may not reflect requests visible when answers were written.','Do not extrapolate reviewed transitions to the unreviewed 183-case cohort.','Review defines overall sufficiency only; no new independent scores for the three Qwen dimensions.']}
save('case_results.json',cases); save('review_comparison.json',summary)
sources=list(paths.values())+[R/'mapping_private.json',R/'freeze_receipt.json',Path(__file__)]
save('analysis_manifest.json',{'source_sha256':{str(p.relative_to(B)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},'review_hashes_verified':True,'output_cases':len(cases)})
print(json.dumps(summary,ensure_ascii=True,indent=2))
