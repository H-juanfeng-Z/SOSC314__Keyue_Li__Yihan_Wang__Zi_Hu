"""Export reproducible decision fields, excluding raw prompts and free-text outputs."""
import argparse
import hashlib
import json
from pathlib import Path

def export(source,target):
    fields=['suite','id','case_id','pair_id','question_id','arm','category','sample_phase','status','input_tokens','output_tokens','seconds','errors']
    rows=[json.loads(s) for s in source.read_text(encoding='utf8').splitlines()]
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('x',encoding='utf8',newline='\n') as f:
        for r in rows:
            item={k:r[k] for k in fields if k in r}
            item['inference_requested']='messages' in r
            if isinstance(r.get('parsed'),dict):
                item['parsed']={k:v for k,v in r['parsed'].items() if k not in ['reason','request_summary']}
            f.write(json.dumps(item,ensure_ascii=False)+'\n')
    print(json.dumps({'records':len(rows),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'export_sha256':hashlib.sha256(target.read_bytes()).hexdigest()}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();export(a.source,a.out)
