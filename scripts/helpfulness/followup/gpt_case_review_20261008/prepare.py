"""Freeze targeted cases and deterministic controls; hide prior labels and rank."""
import json,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent;B=R.parent
def read(p):return json.loads(p.read_text(encoding='utf8'))
def rank(q):return hashlib.sha256(('gpt-case-audit-314:'+str(q)).encode()).hexdigest()
def save(p,x):
    with p.open('x',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,indent=2)
d=read(B/'final_trajectory_20261008/results/question_level.json')
a={r['question_id']:r for r in d['14B/2/substantial/pooled']};b={r['question_id']:r for r in d['7B/2/substantial/pooled']}
added={q for q,r in a.items() if r['rescue']};disagreement={q for q in a.keys()&b.keys() if a[q]['rescue']!=b[q]['rescue']}
target=added|disagreement;controls=set(sorted(set(a)-target,key=rank)[:8]);selected=sorted(target|controls,key=rank)
meta=read(B/'answer_trajectory_20261007/answer_metadata.json');tasks={t['pair_id']:t for t in read(B/'answer_trajectory_20261007/tasks.json')}
pack=[];key=[]
for i,q in enumerate(selected,1):
    ms=sorted([m for m in meta if m['question_id']==q],key=lambda m:rank(m['answer_id']))
    ms=[m for m in ms if m['answer_rank']<=2];assert len(ms)==2
    first=json.loads(tasks[ms[0]['pair_id']]['messages'][1]['content'].split('DATA:\n',1)[1])
    case={'case_id':f'R{i:03}','title':first['title'],'question_lines':{f'Q{j+1}':line for j,line in enumerate(first['question'].splitlines()) if line.strip()},'answers':{}}
    mapping={}
    for letter,m in zip(['X','Y'],ms):
        case['answers'][letter]=tasks[m['pair_id']]['lines'];mapping[letter]=m
    pack.append(case);key.append(dict(case_id=case['case_id'],question_id=q,added=q in added,disagreement=q in disagreement,control=q in controls,answers=mapping))
save(R/'review_packet.json',pack);save(R/'mapping_private.json',key)
save(R/'protocol.json',dict(reviewer='GPT assistant in current Codex conversation; exact backend model ID not exposed to this review script',review_type='AI evidence review, not human gold or strict blinded independent validation',known_context='Reviewer has seen aggregate results and some previous examples. Packet hides previous model scores, author identity and chronological answer rank.',sampling={'added':len(added),'disagreement':len(disagreement),'overlap':len(added&disagreement),'target_union':len(target),'controls':len(controls),'total':len(selected),'salt':'gpt-case-audit-314'},unit='Question with two independently rated answers, shown in hash-randomized order',rubric={'substantial':'Supplied answer adequately addresses central expressed request; narrow correction or conceptual explanation can suffice without complete application.','partial':'Some usable help, but central request has material gaps or unsupported solution claims.','unhelpful':'No substantive usable assistance for expressed request.','uncertain':'Essential missing context or unresolved technical issue prevents defensible classification.'},constraints=['No execution, external-link contents not assumed','Distinguish stated plausibility from verified correctness','Cite supplied answer line IDs','Record ambiguity rather than solve missing content','Targeted sample cannot estimate population accuracy','Eight controls are sampled from non-target remainder, not full population','No revisions after unmasking without an explicit amendment log','No GitHub uploads']))
print(json.dumps(read(R/'protocol.json')['sampling']))
