"""Frozen evidence-dependence diagnostics on 96 reused development pairs."""
import json,hashlib,random,shutil
from pathlib import Path
R=Path(__file__).resolve().parent;OLD=R.parent/'rubric_check_20261007'
def save(n,x):
 with (R/n).open('x',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,indent=2)
prior=[json.loads(s) for s in (OLD/'results/calls.jsonl').read_text(encoding='utf8').splitlines()]
source=json.loads((OLD/'tasks.json').read_text(encoding='utf8'))
orig={t['pair_id']:t for t in source if t['suite']=='original' and t['arm']=='no_examples_control'}
ratings={r['pair_id']:r for r in prior if r['suite']=='original' and r['arm']=='no_examples_control' and r['status']=='valid'}
tasks=[];audit=[]
for pid,t in orig.items():
 prefix,raw=t['messages'][1]['content'].split('DATA:\n',1);data=json.loads(raw);lines=data['ANSWER_LINES'];cited=ratings[pid]['parsed']['evidence_ids'];assert all(k in lines for k in cited)
 rng=random.Random(int(hashlib.sha256((pid+':314').encode()).hexdigest(),16))
 controls=rng.sample(list(lines),len(cited))
 variants={'original':lines,'cited_removed':{k:v for k,v in lines.items() if k not in cited},'random_removed':{k:v for k,v in lines.items() if k not in controls},'cited_only':{k:v for k,v in lines.items() if k in cited}}
 audit.append({'pair_id':pid,'cited_ids':cited,'random_ids':controls,'overlap':len(set(cited)&set(controls)),'line_count':len(lines),'cited_characters':sum(len(lines[k]) for k in cited),'random_characters':sum(len(lines[k]) for k in controls)})
 for arm,ls in variants.items():
  d=dict(data,ANSWER_LINES=ls,answer_present=bool(ls));tasks.append(dict(t,task_id=f'helpfulness/{pid}/{arm}',suite='helpfulness',arm=arm,lines=ls,messages=[t['messages'][0],{'role':'user','content':prefix+'DATA:\n'+json.dumps(d,ensure_ascii=False)}]))
bridge=[json.loads(s) for s in (R.parent/'request_bridge_20260927/results/calls.jsonl').read_text(encoding='utf8').splitlines()]
for r in bridge:
 if r['pair_id'] not in orig or r['suite']!='intent' or r['arm']!='direct':continue
 prefix,raw=r['messages'][1]['content'].split('QUESTION LINES:\n',1);lines=json.loads(raw)
 for arm,ls in [('full',lines),('title_only',{'T':lines['T']})]:
  tasks.append({'task_id':f"intent_{r['category']}/{r['pair_id']}/{arm}",'pair_id':r['pair_id'],'question_id':r['question_id'],'suite':'intent_'+r['category'],'arm':arm,'validator':'intent','lines':ls,'messages':[r['messages'][0],{'role':'user','content':prefix+'QUESTION LINES:\n'+json.dumps(ls,ensure_ascii=False)}]})
tasks.sort(key=lambda t:(hashlib.sha256((t['pair_id']+':314').encode()).hexdigest(),t['suite'],t['arm']))
assert len(tasks)==1728 and len({t['task_id'] for t in tasks})==1728
save('tasks.json',tasks);save('audit.json',audit);save('prior_controls.json',list(ratings.values()))
save('manifest.json',{'planned':1728,'pairs':96,'tasks_sha256':hashlib.sha256((R/'tasks.json').read_bytes()).hexdigest(),'reserved_used':False})
save('plan.json',{'purpose':'Test whether intent depends on question-body evidence and helpfulness ratings depend on cited answer content.','design':'96 reused pairs:7 intent categories x2 input views +4 answer variants =1728 generations.','intent':'Identical direct intent prompt, full question versus title only. Missing context may legitimately yield uncertain; no accuracy claim. Full control compared with old labels.', 'helpfulness':'Fixed no-example rubric. Original, cited-lines removed, equal-number random-lines removed, cited-lines only. Original IDs preserved. Random selection may overlap citations; audit overlap and character counts.', 'limitations':['Citations are model-selected, not causal ground truth','Deleting lines can break code syntax, remove context or leave redundant assistance; changes are exploratory and not expected always to lower scores','Random deletion matches line count, not length or semantic importance','All96 are reused development cases; no reserved80','Same model cannot independently validate itself'], 'analysis':'Separate parse validity from semantic constraints; paired transitions, absence of evidence for positive ratings, control reproduction. No automatic relabeling or selecting favorable associations.','budget':'One GPU, four CPU threads, max3hours. Stop at fixed task count, no automatic expansion.','next':'Review high-impact evidence failures, freeze protocol and then sensitivity-adjusted association analysis.'})
for n in ['run.py','launch_newserver.py','request_protocol.py','run_helpfulness_grounding.py']:shutil.copyfile(OLD/n,R/n)
print('Frozen',len(tasks),'tasks')
