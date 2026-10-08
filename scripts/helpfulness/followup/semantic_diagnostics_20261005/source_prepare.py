"""Freeze selected development diagnostics, preserving all earlier records."""
import collections, hashlib, json, sys
from pathlib import Path
R=Path(__file__).resolve().parent
W=R.parents[1]
sys.path.insert(0,str(W/'SOSC314__Keyue_Li__Yihan_Wang__Zi_Hu/scripts/helpfulness'))
import run_helpfulness_grounding as base
OLD=W/'sosc314_week4_experiments/measurement_expansion_20260930'
def read(p): return json.loads(p.read_text(encoding='utf8'))
def save(name,obj):
    with (R/name).open('x',encoding='utf8') as f: json.dump(obj,f,ensure_ascii=False,indent=2)
def rank(ids,salt): return sorted(ids,key=lambda p:hashlib.sha256((p+salt).encode()).hexdigest())
rows=[json.loads(s) for s in (OLD/'results/calls.jsonl').read_text(encoding='utf8').splitlines()]
idx={(r['pair_id'],r['suite'],r['arm']):r for r in rows}
cases={c['pair_id']:c for c in read(W/'sosc314_week4_experiments/request_bridge_20260927/data/pairs.json')}
oldsel=read(OLD/'selection_private.json'); corrected=[]
for s in oldsel:
    s=dict(s);suite,pid,arm=s['task_id'].split('/')
    if suite=='intent':
        a=idx[pid,suite,'direct_control']['parsed']['decision'];b=idx[pid,suite,'extraction_control']['parsed']['decision']
        s['prior_selection_group']=s['selection_group']
        s['selection_group']=s['selection_group'].split('/')[0]+'/'+('unchanged' if a==b else a+'_to_'+b)
    corrected.append(s)
save('corrected_selection.json',corrected)
changes=[];stable=[]
for pid in cases:
    a=idx.get((pid,'helpfulness','joint_control'));b=idx.get((pid,'helpfulness','joint_no_examples'))
    if a and b and a['status']==b['status']=='valid':
        (stable if a['parsed']['overall']==b['parsed']['overall'] else changes).append(pid)
selected=rank(changes,':semantic:314')[:24]+rank(stable,':semantic:314')[:24]
assert len(selected)==48
# Balanced synthetic format demonstrations, not labeled natural-data examples.
examples=[
 {'relevance':0,'actionability':0,'sufficiency':0,'overall':'unhelpful','evidence_ids':[],'reason':'The supplied content offers no help with the request.'},
 {'relevance':2,'actionability':2,'sufficiency':1,'overall':'partial','evidence_ids':['EXAMPLE_ONLY'],'reason':'A usable step is provided but an essential requested part is missing.'},
 {'relevance':2,'actionability':2,'sufficiency':2,'overall':'substantial','evidence_ids':['EXAMPLE_ONLY'],'reason':'The answer supplies the assistance needed for the expressed request.'},
 {'relevance':None,'actionability':None,'sufficiency':None,'overall':'uncertain','evidence_ids':[],'reason':'Essential context is missing, preventing a defensible assessment.'}]
def demo(items): return 'Output format examples for DIFFERENT hypothetical cases, not judgments of this case:\n'+'\n'.join(json.dumps(e) for e in items)+'\nNever copy EXAMPLE_ONLY. Use only supplied line IDs for the actual case.\n'
tasks=[];audit=[]
for n,pid in enumerate(selected):
    c=cases[pid];prior=idx[pid,'helpfulness','joint_control'];msg=prior['messages'];prefix=msg[1]['content'].split('DATA:\n',1)[0]
    assert base.EXTRA in prefix
    donor=cases[selected[(n+24)%48]]
    conditions={'original':c['answer_text'],'empty':'','generic':'That is an interesting question. I hope somebody can help you.','swapped':donor['answer_text']}
    prompts={'original_examples':prefix,'no_examples':prefix.replace(base.EXTRA,'',1),'balanced':prefix.replace(base.EXTRA,demo(examples),1),'balanced_reverse':prefix.replace(base.EXTRA,demo(list(reversed(examples))),1)}
    audit.append({'pair_id':pid,'question_id':c['question_id'],'answer_id':c['answer_id'],'stratum':'changed' if pid in changes else 'unchanged','title':c['title'],'question_text':c['question_text'],'answer_text':c['answer_text'],'prior_control':prior['parsed'],'prior_no_examples':idx[pid,'helpfulness','joint_no_examples']['parsed'],'donor_pair_id':donor['pair_id']})
    for condition,answer in conditions.items():
        lines={f'A{i+1}':s for i,s in enumerate(answer.splitlines()) if s.strip()}
        data={'title':c['title'],'question':c['question_text'],'answer_present':bool(lines),'ANSWER_LINES':lines}
        for name,prompt in prompts.items():
            messages=[msg[0],{'role':'user','content':prompt+'DATA:\n'+json.dumps(data,ensure_ascii=False)}]
            if condition=='original' and name=='original_examples': assert messages==msg
            tasks.append({'task_id':f'{condition}/{pid}/{name}','pair_id':pid,'question_id':c['question_id'],'suite':condition,'arm':name,'messages':messages,'validator':'joint','lines':lines})
tasks.sort(key=lambda t:(hashlib.sha256((t['pair_id']+':314').encode()).hexdigest(),t['suite'],t['arm']))
assert len(tasks)==768 and len({t['task_id'] for t in tasks})==768
save('tasks.json',tasks);save('case_audit.json',audit)
intent_ids=rank([p for p in cases if (p,'intent','summary_only') in idx and idx[p,'intent','direct_control']['parsed']['decision']!=idx[p,'intent','summary_only']['parsed']['decision']],':intent-audit:314')[:24]
save('intent_audit.json',[{'case':cases[p],'decisions':{a:idx[p,'intent',a]['parsed'] for a in ['direct_control','extraction_control','summary_only']}} for p in intent_ids])
save('manifest.json',{'planned':768,'cases':48,'changed':24,'unchanged':24,'tasks_sha256':hashlib.sha256((R/'tasks.json').read_bytes()).hexdigest(),'source_log_sha256':hashlib.sha256((OLD/'results/calls.jsonl').read_bytes()).hexdigest(),'reserved_used':False,'corrected_intent_strata':dict(collections.Counter(s['selection_group'] for s in corrected if s['task_id'].startswith('intent/')))})
save('plan.json',{'purpose':'Cross answer-content controls with prompt examples; no accuracy claims.','model':'Unchanged Qwen2.5-Coder-14B greedy fp16','design':'48 selected reused cases x4 answer conditions x4 prompt conditions','expectations':{'empty':'all dimension scores0 and overall unhelpful','generic':'no substantive assistance; all scores0 and overall unhelpful','swapped':'exploratory only; donor answers may accidentally help, not gold negatives','original':'no known correct rating'},'interpretation':'Balanced versus absent examples changes content and length; reverse order isolates order within the same example set. Neither high scores nor agreement selects a winner. Sample is outcome-stratified, not prevalence representative. No partial deletions or invented complete answers without case review.','next':'Review actual disagreement cases and freeze measurement protocol before linking intent and helpfulness.','budget':'One scheduledGPU,4CPU,64GiB,3h cap; roughly1-2h expected. No GitHub upload. Preserve held64773.'})
print('Frozen768 tasks; corrected labels in separate file; originals unchanged.')
