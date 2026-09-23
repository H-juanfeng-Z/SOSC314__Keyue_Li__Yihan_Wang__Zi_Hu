"""Four-cell input/encoder ablation; exploratory response outcome, not quality.

Use on allocated compute node only. Never executes source-post code.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

def main():
    p=argparse.ArgumentParser()
    for k in ['data','minilm','codebert','out']: p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--n',type=int,default=10000); p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--threads',type=int,default=8)
    p.add_argument('--profile-batches',action='store_true')
    a=p.parse_args(); a.out.mkdir(parents=True,exist_ok=True); torch.set_num_threads(a.threads)
    df=pd.read_json(a.data,lines=True)
    # Stable question-ID selection, no response or score-based selection.
    df['selection_key']=df.question_id.map(lambda q:hashlib.sha256(f'{q}:w4encoder:314'.encode()).hexdigest())
    df=df.sort_values('selection_key').head(a.n).reset_index(drop=True)
    df[['question_id','creation_year']].to_json(a.out/'sample_ids.json',orient='records',indent=2)
    prose=df.title.fillna('')+'\n'+df.natural_language_text.fillna('')+'\nTAGS '+df.tags.fillna('')
    code=df.code_text.fillna('')
    masks={'train':df.creation_year.le(2014).to_numpy(),'validation':df.creation_year.eq(2015).to_numpy(),'test':df.creation_year.eq(2016).to_numpy()}
    y=df.answered_within_24h.to_numpy()
    results={'data_sha256':hashlib.sha256(a.data.read_bytes()).hexdigest(),'n':len(df),'outcome':'answer within 24h, NOT helpfulness',
        'policy':'Frozen encoder; masked mean pooling + L2 normalization; balanced LR C=1; no tuning.',
        'caveat':'Both use 512 token limit but different tokenizers. Paired input uses longest_first truncation, so code can displace prose. CodeBERT pooling is a baseline, not semantic correctness verification.',
        'splits':{k:{'n':int(v.sum()),'positive_rate':float(y[v].mean())} for k,v in masks.items()},'models':[]}
    device='cuda' if torch.cuda.is_available() else 'cpu'
    for name,path in [('minilm',a.minilm),('codebert',a.codebert)]:
        tok=AutoTokenizer.from_pretrained(path,local_files_only=True,trust_remote_code=False)
        model=AutoModel.from_pretrained(path,local_files_only=True,trust_remote_code=False,use_safetensors=True).eval().to(device)
        profile=[]
        chosen_batch=a.batch_size
        if a.profile_batches and device=='cuda':
            sample=prose.loc[masks['train']].iloc[:256].tolist()
            sample_code=code.loc[masks['train']].iloc[:256].tolist()
            for bs in [16,32,64]:
                trials=[]
                try:
                    torch.cuda.reset_peak_memory_stats()
                    for repeat in range(3):
                        torch.cuda.synchronize(); start_profile=time.perf_counter()
                        with torch.inference_mode():
                            for offset in range(0,len(sample),bs):
                                e=tok(sample[offset:offset+bs],text_pair=sample_code[offset:offset+bs],padding=True,truncation=True,max_length=512,return_tensors='pt').to(device)
                                h=model(**e).last_hidden_state
                        torch.cuda.synchronize(); elapsed=time.perf_counter()-start_profile
                        if repeat>0: trials.append(elapsed)
                    profile.append({'batch_size':bs,'median_seconds':float(np.median(trials)), 'peak_allocated_bytes':torch.cuda.max_memory_allocated()})
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache(); profile.append({'batch_size':bs,'error':'out_of_memory'})
            successful=[v for v in profile if 'median_seconds' in v]
            if not successful: raise RuntimeError('No profile batch fits GPU')
            chosen_batch=min(successful,key=lambda v:v['median_seconds'])['batch_size']
            (a.out/(name+'_batch_profile.json')).write_text(json.dumps(profile,indent=2))
            del h,e
        for with_code in [False,True]:
            name_run=name+('_prose_code' if with_code else '_prose')
            vec=[]; clipped=0; start=time.perf_counter()
            with torch.inference_mode():
                for i in range(0,len(df),chosen_batch):
                    batch=prose.iloc[i:i+chosen_batch].tolist()
                    pair=code.iloc[i:i+chosen_batch].tolist() if with_code else None
                    raw=tok(batch,text_pair=pair,truncation=False)['input_ids']
                    clipped+=sum(len(x)>512 for x in raw)
                    enc=tok(batch,text_pair=pair,padding=True,truncation='longest_first',max_length=512,return_tensors='pt').to(device)
                    h=model(**enc).last_hidden_state; m=enc.attention_mask.unsqueeze(-1)
                    v=(h*m).sum(1)/m.sum(1).clamp(min=1)
                    vec.append(torch.nn.functional.normalize(v,p=2,dim=1).cpu().numpy())
                    if i%(chosen_batch*50)==0: print(name_run,i,len(df),flush=True)
            x=np.concatenate(vec); np.save(a.out/(name_run+'.npy'),x)
            clf=LogisticRegression(max_iter=2000,C=1,class_weight='balanced').fit(x[masks['train']],y[masks['train']])
            row={'name':name_run,'dimensions':x.shape[1],'truncated':clipped,'seconds':time.perf_counter()-start,'device':device,'batch_size':chosen_batch,'splits':{}}
            for split in ['validation','test']:
                mask=masks[split]; pred=clf.predict_proba(x[mask])[:,1]
                row['splits'][split]={'roc_auc':roc_auc_score(y[mask],pred),'average_precision':average_precision_score(y[mask],pred)}
                np.savez_compressed(a.out/(name_run+'_'+split+'.npz'),question_id=df.question_id.to_numpy()[mask],y=y[mask],probability=pred)
            results['models'].append(row)
            (a.out/'results.json').write_text(json.dumps(results,indent=2))
            print(json.dumps(row),flush=True)
        del model
        if device=='cuda': torch.cuda.empty_cache()

if __name__=='__main__': main()
