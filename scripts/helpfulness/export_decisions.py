"""Remove source text and generated rationales while retaining analysis fields."""
import argparse
import hashlib
import json
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    keys=['arm','case_id','source','category','status','variant','pair_id','condition',
          'donor_pair','answer_sha256','errors','cites_marker_only_line']
    parsed_keys=['decision','relevance','actionability','sufficiency','overall']
    a.output.parent.mkdir(parents=True,exist_ok=True)
    count=0
    with a.output.open('x',encoding='utf8') as out:
        for line in a.input.open(encoding='utf8'):
            r=json.loads(line)
            record={k:r[k] for k in keys if k in r}
            if isinstance(r.get('parsed'),dict):
                record['parsed']={k:v for k,v in r['parsed'].items() if k in parsed_keys}
            out.write(json.dumps(record)+'\n')
            count+=1
    manifest={'source_sha256':hashlib.sha256(a.input.read_bytes()).hexdigest(),
              'export_sha256':hashlib.sha256(a.output.read_bytes()).hexdigest(),'rows':count}
    a.output.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')

if __name__=='__main__':main()
