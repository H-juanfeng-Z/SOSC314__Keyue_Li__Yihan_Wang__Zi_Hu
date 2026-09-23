"""Bounded post-job analysis of JSON results and telemetry logs; no model runs."""
from pathlib import Path
import argparse
import json
import csv
from collections import Counter
import numpy as np
from sklearn.metrics import roc_auc_score

p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); a=p.parse_args()
r=a.root; result={'labels':{},'encoders':None,'gpu_telemetry':{},'cross_model_agreement':{}}
annotations={}
for size in ['15b','3b','7b']:
    folder=r/'results'/('helpfulness_'+size)
    if not (folder/'summary.json').exists(): result['labels'][size]={'status':'missing_or_incomplete'}; continue
    s=json.loads((folder/'summary.json').read_text()); rows=json.loads((folder/'annotations.json').read_text())
    s['status_counts']=dict(Counter(v['status'] for v in rows))
    original=[v for v in rows if not v['control'] and v['status']=='valid']
    s['original_overall_distribution']=dict(Counter(v['parsed']['overall'] for v in original))
    s['original_rating_distributions']={key:dict(Counter(str(v['parsed'][key]) for v in original)) for key in ['relevance','actionability','sufficiency']}
    result['labels'][size]=s
    annotations[size]={(v['pair_id'],v['variant']):v['parsed']['overall'] for v in original}
for left,right in [('15b','3b'),('15b','7b'),('3b','7b')]:
    if left in annotations and right in annotations:
        common=annotations[left].keys()&annotations[right].keys()
        result['cross_model_agreement'][left+'_'+right]={'n':len(common),'agreement':sum(annotations[left][k]==annotations[right][k] for k in common)/len(common) if common else None}
folder=r/'results/encoder_60000'
if (folder/'results.json').exists():
    result['encoders']=json.loads((folder/'results.json').read_text())
    names=['minilm_prose','minilm_prose_code','codebert_prose','codebert_prose_code']
    if all((folder/(n+'_test.npz')).exists() for n in names):
        preds={n:np.load(folder/(n+'_test.npz')) for n in names}; ref=preds[names[0]]; y=ref['y']
        assert all(np.array_equal(v['question_id'],ref['question_id']) and np.array_equal(v['y'],y) for v in preds.values())
        pairs=[('minilm_prose_code','minilm_prose'),('codebert_prose_code','codebert_prose'),('minilm_prose','codebert_prose_code')]
        samples={a+' minus '+b:[] for a,b in pairs}; rng=np.random.default_rng(314)
        for _ in range(1000):
            ix=rng.integers(0,len(y),len(y)); scores={n:roc_auc_score(y[ix],v['probability'][ix]) for n,v in preds.items()}
            for x,z in pairs: samples[x+' minus '+z].append(scores[x]-scores[z])
        result['paired_auc_differences']={x+' minus '+z:{'delta':roc_auc_score(y,preds[x]['probability'])-roc_auc_score(y,preds[z]['probability']),
             'percentile95':np.quantile(samples[x+' minus '+z],[.025,.975]).tolist()} for x,z in pairs}
for f in (r/'logs').glob('gpu-*.csv'):
    # nvidia-smi telemetry, not a user spreadsheet; failures left explicit.
    try:
        rows=list(csv.reader(f.open())); nums=[]
        for row in rows[1:]:
            if len(row)==6: nums.append([float(row[k].strip().split()[0]) for k in [2,4]])
        v=np.array(nums)
        result['gpu_telemetry'][f.stem]={'samples':len(v),'mean_util_percent':float(v[:,0].mean()),'p95_util_percent':float(np.quantile(v[:,0],.95)),
            'peak_used_mib':float(v[:,1].max()),'share_samples_above80pct':float((v[:,0]>=80).mean()),
            'note':'Includes loading and CPU stages; used memory includes framework/driver overhead, not only tensors.'}
    except Exception as exc: result['gpu_telemetry'][f.stem]={'error':str(exc)}
result['caveats']=['No human annotation accuracy; matching answer-line IDs does not verify the evidence supports the rating.',
    'Cross-model and cross-prompt agreement are not independent ground truth.',
    'Mismatch controls are diagnostic, not representative negative samples.',
    'Question bootstrap conditional on fitted models; reused temporal test set, no causal claims.']
(r/'results/summary_all.json').write_text(json.dumps(result,indent=2))
lines=['# Round 2 experiment summary','', 'No human accuracy estimates. See summary_all.json for complete diagnostics.','']
for size,s in result['labels'].items():
    lines.append(f"- {size}: valid {s.get('valid','NA')}/{s.get('calls','NA')}; prompt agreement {s.get('original_pair_prompt_agreement','NA')}; mismatched controls relevance=0: {s.get('control_relevance_zero','NA')}/{s.get('control_calls_valid','NA')}")
if result['encoders']:
    lines+=['','| Model | Test AUC | Test AP | Batch |','|---|---:|---:|---:|']
    for m in result['encoders']['models']:
        t=m['splits']['test']; lines.append(f"|{m['name']}|{t['roc_auc']:.6f}|{t['average_precision']:.6f}|{m.get('batch_size')}|")
lines+=['','## GPU telemetry','',json.dumps(result['gpu_telemetry'],indent=2),'','## Caveats','']+result['caveats']
(r/'results/SUMMARY.md').write_text('\n'.join(lines))
print(json.dumps(result,indent=2))
