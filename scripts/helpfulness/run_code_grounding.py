"""Development-only content grounding: preserve prose while exchanging code."""
import hashlib
import json
import re
import sys
from pathlib import Path

BLOCK=re.compile(r'\[CODE\]\s*\n?(.*?)\n?\s*\[/CODE\]',re.S)

def trials(cases):
    codes=['\n\n'.join(BLOCK.findall(c['answer_text'])).strip() for c in cases]
    for i,c in enumerate(cases):
        if not codes[i]:
            continue
        donors=[j for j in range(len(cases)) if j!=i and codes[j] and codes[j]!=codes[i]]
        if not donors:raise ValueError('No different development donor code')
        # Deterministic nearest length; no model scores used in selection.
        j=min(donors,key=lambda j:(abs(len(codes[j])-len(codes[i])),cases[j]['pair_id']))
        parts=BLOCK.split(c['answer_text'])
        # Insert all donor code in first marked block; later code blocks removed.
        swapped=parts[0]+'[CODE]\n'+codes[j]+'\n[/CODE]'+''.join(parts[2::2])
        variants=[('original',c['answer_text']),('swapped_code',swapped),('code_only',codes[i]),('foreign_code_only',codes[j])]
        for name,text in variants:
            yield c,name,text,cases[j]['pair_id'] if name in ['swapped_code','foreign_code_only'] else None

def main():
    import run_helpfulness_grounding as base
    out=Path(sys.argv[sys.argv.index('--out')+1]);out.mkdir(parents=True,exist_ok=True)
    cases=json.loads(Path(sys.argv[sys.argv.index('--data')+1]).read_text(encoding='utf8'))[:20]
    rows=list(trials(cases))
    with (out/'protocol.json').open('x') as f:
        json.dump({'expected_calls':len(rows)*2,'original_development_cases':20,'eligible_cases':len(rows)//4,'reserved_used':False,'donor_rule':'different case, different nonempty marked code, nearest character length then pair_id','interpretation':'Donor code not guaranteed irrelevant. No accuracy or automatic lower-quality labels. Swapping can change code length and block structure. Text outside original code blocks is retained byte-for-byte.','trials':[{'pair_id':c['pair_id'],'condition':name,'donor_pair':donor,'answer_text':text,'chars':len(text),'sha256':hashlib.sha256(text.encode()).hexdigest()} for c,name,text,donor in rows]},f,indent=2)
    base.conditions=trials
    # Verify that the existing runner records the filtered trial count.
    import inspect
    source=inspect.getsource(base.main)
    old="expected_calls=len(trials)*2,control_suite=a.control_suite"
    assert old in source
    # Reuse the runner unchanged; it overrides its default count dynamically.
    base.main()

if __name__=='__main__':main()
