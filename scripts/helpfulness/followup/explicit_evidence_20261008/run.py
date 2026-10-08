"""Fixed-size diagnostics with incremental logs; no model-generated code execution."""
import argparse,collections,hashlib,json,time
from pathlib import Path
import run_helpfulness_grounding as base
import request_protocol

def validate(obj,task):
 if task['validator']=='joint':return base.validate(obj,task['lines'])
 if task['validator']=='intent':return request_protocol.validate_intent(obj,task['lines'])
 if not isinstance(obj,dict):return ['not_object']
 errors=[]
 if 'score' not in obj or (obj['score'] is not None and (type(obj['score']) is not int or obj['score'] not in [0,1,2])):errors.append('score')
 ids=obj.get('evidence_ids')
 if not isinstance(ids,list) or len(ids)>3 or any(not isinstance(x,str) or x not in task['lines'] for x in ids):errors.append('evidence_ids')
 if not isinstance(obj.get('reason'),str) or not obj['reason'].strip():errors.append('reason')
 return errors

def main():
 import torch,transformers
 from transformers import AutoTokenizer,AutoModelForCausalLM
 p=argparse.ArgumentParser()
 for k in ['data','model','out']:p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 assert not (a.out/'calls.jsonl').exists(),'Refuse overwrite or automatic resume'
 tasks=json.loads(a.data.read_text(encoding='utf8'));assert len(tasks)>0
 torch.set_num_threads(4);torch.manual_seed(314);assert torch.cuda.is_available()
 free,total=torch.cuda.mem_get_info()
 preflight={'free_bytes':free,'total_bytes':total,'device_name':torch.cuda.get_device_name(0),'device_count':torch.cuda.device_count(),'minimum_free_bytes':36*1024**3}
 (a.out/'preflight.json').write_text(json.dumps(preflight,indent=2),encoding='utf8')
 print('GPU_PREFLIGHT',json.dumps(preflight),flush=True)
 assert free>36*1024**3,'Allocated GPU lacks free memory; do not bypass this check'
 started=time.time();config={'planned':len(tasks),'seed':314,'model':str(a.model),'torch':torch.__version__,'transformers':transformers.__version__,'max_input':7500,'max_output':256,'do_sample':False,'input_sha256':hashlib.sha256(a.data.read_bytes()).hexdigest(),'scripts_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob('*.py')}}
 (a.out/'manifest.json').write_text(json.dumps(config,indent=2),encoding='utf8')
 tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True,trust_remote_code=False)
 model=AutoModelForCausalLM.from_pretrained(a.model,local_files_only=True,trust_remote_code=False,use_safetensors=True,torch_dtype=torch.float16).eval().to('cuda')
 counts=collections.Counter()
 with (a.out/'calls.jsonl').open('x',encoding='utf8') as f:
  for i,t in enumerate(tasks):
   if time.time()-started>2.75*3600 and (i==0 or (t['suite'],t['pair_id'])!=(tasks[i-1]['suite'],tasks[i-1]['pair_id'])):break
   r={k:t[k] for k in ['task_id','pair_id','question_id','suite','arm','messages']}
   inp=tok(tok.apply_chat_template(t['messages'],tokenize=False,add_generation_prompt=True),return_tensors='pt',truncation=False).to('cuda');n=inp.input_ids.shape[1];r['input_tokens']=n
   if n>7500:r['status']='context_skipped'
   else:
    start=time.perf_counter()
    with torch.inference_mode():output=model.generate(**inp,do_sample=False,max_new_tokens=256,pad_token_id=tok.eos_token_id)
    raw=tok.decode(output[0,n:],skip_special_tokens=True);r.update(raw=raw,seconds=time.perf_counter()-start,output_tokens=int(output.shape[1]-n))
    try:
     obj=json.loads(raw[raw.index('{'):raw.rindex('}')+1]);errors=validate(obj,t);r.update(parsed=obj,errors=errors,status='invalid' if errors else 'valid')
    except (ValueError,TypeError,KeyError):r['status']='invalid_json'
   counts[t['suite']+'/'+t['arm']+'/'+r['status']]+=1
   f.write(json.dumps(r,ensure_ascii=False)+'\n');f.flush()
   progress={'completed_calls':i+1,'planned':len(tasks),'statuses':dict(counts),'elapsed_seconds':time.time()-started}
   (a.out/'progress.json').write_text(json.dumps(progress,indent=2),encoding='utf8')
   print(i+1,t['task_id'],r['status'],flush=True)
 progress['completed']=progress['completed_calls']==len(tasks);progress['stop_reason']='all_tasks' if progress['completed'] else 'time_budget';(a.out/'summary.json').write_text(json.dumps(progress,indent=2),encoding='utf8')
if __name__=='__main__':main()
