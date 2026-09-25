"""Same-model paired answer-content sensitivity, development20 only."""
import json
import re
from pathlib import Path

BLOCK = re.compile(r'\[CODE\]\s*\n?(.*?)\n?\s*\[/CODE\]', re.S)

def transform(answer):
    blocks=BLOCK.findall(answer)
    prose=BLOCK.sub('', answer).strip()
    code='\n\n'.join(blocks).strip()
    # Deterministic character prefix; may cut a sentence/code block. Not a gold inferior answer.
    prefix=answer[:(len(answer)+1)//2]
    return [('original',answer),('prose_only',prose),('code_only',code),('prefix_half',prefix)]

def conditions(cases):
    for c in cases:
        for name,answer in transform(c['answer_text']):
            yield c,name,answer,None

def main():
    import sys
    import run_helpfulness_grounding as base
    assert '--control-suite' not in sys.argv
    out=Path(sys.argv[sys.argv.index('--out')+1]); out.mkdir(parents=True,exist_ok=True)
    data=Path(sys.argv[sys.argv.index('--data')+1])
    cases=json.loads(data.read_text(encoding='utf8'))[:20]
    with (out/'transformation_manifest.json').open('x') as f:
        json.dump({'reserved_used':False,'code_definition':'Only existing [CODE] markers originating from HTML pre blocks; inline/unmarked code is not removed.','prefix_definition':'First ceil(characters/2) characters; not semantic removal.','interpretation':'Sensitivity only, no presumed correctness or monotonic quality ordering. Empty and unchanged variants analyzed separately.','cases':[{'pair_id':c['pair_id'],'has_marked_code':bool(BLOCK.search(c['answer_text'])),'variants':[{'condition':name,'chars':len(text),'unchanged':text==c['answer_text'],'empty':not text.strip(),'answer_text':text} for name,text in transform(c['answer_text'])]} for c in cases]},f,indent=2)
    base.conditions=conditions
    base.main()

if __name__=='__main__':main()
