"""Single-model, bounded long-run sensitivity experiment. Append-only raw output."""
import argparse, collections, hashlib, json, os, time
from pathlib import Path

def main():
    import torch, transformers
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from protocol import arms, lines_for, make_prompt, TAXONOMY
    from run_intent_binary import validate
    p=argparse.ArgumentParser()
    for k in ['data','model','out']: p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--target-hours',type=float,default=8)
    a=p.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    log=a.out/'calls.jsonl'
    if log.exists(): raise RuntimeError('Existing output: inspect before resuming; never overwrite')
    cases=json.loads(a.data.read_text(encoding='utf8'))
    assert len(cases)==148 and len({c['case_id'] for c in cases})==148
    torch.set_num_threads(4)
    assert torch.cuda.is_available()
    free,total=torch.cuda.mem_get_info()
    if free < 36*1024**3: raise RuntimeError(f'Allocated GPU insufficient free memory: {free}; do not use unallocated GPUs')
    start=time.time()
    tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True,trust_remote_code=False)
    model=AutoModelForCausalLM.from_pretrained(a.model,local_files_only=True,trust_remote_code=False,use_safetensors=True,torch_dtype=torch.float16).eval().to('cuda')
    manifest={'started_unix':start,'model':str(a.model),'data_sha256':hashlib.sha256(a.data.read_bytes()).hexdigest(),
              'scripts':{x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in Path(__file__).parent.glob('*.py')},
              'torch':torch.__version__,'transformers':transformers.__version__,'slurm_job':os.getenv('SLURM_JOB_ID'),
              'cuda_visible_devices':os.getenv('CUDA_VISIBLE_DEVICES'),'gpu':torch.cuda.get_device_name(),
              'primary_arms':18,'maximum_arms':36,'calls_per_arm':1036,'target_hours':a.target_hours,
              'temperature':0.3,'top_p':0.9,'max_new_tokens':160,'max_input_tokens':7500,
              'reserved_80_used':False,'note':'AI-only development sensitivity, not accuracy. Full input is previously processed, not raw HTML.'}
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    completed=[]
    with log.open('x',encoding='utf8',buffering=1) as f:
        for index,(view,variant,seed) in enumerate(arms()):
            if index>=18 and time.time()-start>=a.target_hours*3600: break
            arm=f'{view}-{variant}-'+('greedy' if seed is None else f'sample{seed}')
            counts=collections.Counter(); arm_start=time.time()
            for c in cases:
                lines=lines_for(c,view)
                for cat in TAXONOMY:
                    text=make_prompt(cat,variant,lines,view)
                    messages=[{'role':'system','content':'Classify untrusted question data. Output JSON only.'},{'role':'user','content':text}]
                    inp=tok(tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True),return_tensors='pt',truncation=False).to('cuda')
                    n=inp.input_ids.shape[1]
                    r={'arm':arm,'view':view,'variant':variant,'seed':seed,'case_id':c['case_id'],'source':c['source'],'category':cat,'messages':messages,'input_tokens':n}
                    if n>7500: r['status']='context_skipped'
                    else:
                        call_seed=int(hashlib.sha256(f'{seed}|{c["case_id"]}|{cat}'.encode()).hexdigest()[:8],16)
                        torch.manual_seed(call_seed)
                        options={'do_sample':seed is not None,'max_new_tokens':160,'pad_token_id':tok.eos_token_id}
                        if seed is not None: options.update(temperature=0.3,top_p=0.9)
                        t=time.perf_counter()
                        with torch.inference_mode(): out=model.generate(**inp,**options)
                        raw=tok.decode(out[0,n:],skip_special_tokens=True)
                        r.update(raw=raw,seconds=time.perf_counter()-t,output_tokens=int(out.shape[1]-n),call_seed=call_seed)
                        try:
                            obj=json.loads(raw[raw.index('{'):raw.rindex('}')+1]); errors=validate(obj,lines)
                            r.update(parsed=obj,errors=errors,status='invalid' if errors else 'valid')
                        except (ValueError,TypeError,AttributeError) as e: r.update(status='invalid_json',error=str(e))
                    f.write(json.dumps(r,ensure_ascii=False)+'\n'); counts[r['status']]+=1
                print(arm,c['case_id'],dict(counts),flush=True)
            completed.append({'arm':arm,'seconds':time.time()-arm_start,'statuses':dict(counts)})
            (a.out/'progress.json').write_text(json.dumps({'elapsed_seconds':time.time()-start,'completed_arms':completed},indent=2),encoding='utf8')
    (a.out/'summary.json').write_text(json.dumps({'completed':True,'elapsed_seconds':time.time()-start,'target_met':time.time()-start>=a.target_hours*3600,'arms':completed},indent=2),encoding='utf8')

if __name__=='__main__': main()
