"""Fixed paired diagnostic: reused cases, synthetic checks, unchanged validators."""
import copy, hashlib, json, shutil
from pathlib import Path
R=Path(__file__).resolve().parent
OLD=R.parent/'integrated_20261008'
def read(p): return json.loads(p.read_text(encoding='utf8'))
def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf8') as f: json.dump(x,f,ensure_ascii=False,indent=2)
def rank(x): return hashlib.sha256(('explicit-check-314:'+x).encode()).hexdigest()
features=['environment','constraints','actual_expected']
original=read(OLD/'presentation14/tasks.json')
ids=sorted({t['pair_id'] for t in original},key=rank)[:96]
definitions=read(OLD/'feature_definitions.json')
clarifications={
 'environment':'A named library or runtime without a version is score1, not0. A stated version or platform is score2. Do not infer a version/platform from imports, syntax, or likely defaults. An import explicitly names its library and can support score1.',
 'constraints':'A desired task alone is not a constraint. Score0 if no restriction is stated; score1 for a broad preference such as fast/simple; score2 for an explicit restriction, numeric target, or compatibility requirement. Do not invent performance or compatibility needs.',
 'actual_expected':'An intended output plus unexecuted code is NOT an observed mismatch. Score0 unless the author explicitly reports an actual-versus-desired mismatch. Score1 if a mismatch is explicitly reported but only one side is clear. Score2 requires BOTH observed and desired behaviors in the supplied text. Do not mentally execute code to invent observed output.'}
def make(t,arm):
    t=copy.deepcopy(t);t['arm']=arm;t['task_id']=t['task_id'].rsplit('/',1)[0]+'/'+arm
    if arm=='explicit_boundary':
        name=t['suite'].removeprefix('presentation_')
        text=t['messages'][1]['content'];a,b=text.split('QUESTION LINES:\n',1)
        t['messages'][1]['content']=a+'\nOperational boundary: '+clarifications[name]+'\nReadable absence must be0, not null. Cite only lines that directly support the assigned level.\nQUESTION LINES:\n'+b
    return t
natural=[make(t,a) for t in original if t['pair_id'] in ids and t['suite'].removeprefix('presentation_') in features for a in ['baseline','explicit_boundary']]
# Designed diagnostic expectations, not independent annotation of natural data.
controls={
 'environment':[(0,'How can I sort these values?'),(1,'How can I sort a pandas DataFrame?'),(2,'I use pandas version2.2 on Linux. How can I sort a DataFrame?'),(0,'How do I display the result?'),(1,'I use Python. How do I display the result?'),(2,'I use Python3.11 on Windows11. How do I display the result?')],
 'constraints':[(0,'How can I sort a list?'),(1,'How can I sort a list quickly?'),(2,'How can I sort a list without allocating a second list?'),(0,'How can I read a file?'),(1,'I want a simple way to read a file.'),(2,'How can I read a file while using at most10MB of memory?')],
 'actual_expected':[(0,'I want the output to be6. My code is print(2+3).'),(1,'My program produces the wrong result:5.'),(2,'My program produces5, but I expected6.'),(0,'How can I display a sorted list?'),(1,'The output is wrong; I expected a sorted list.'),(2,'The program displays[3,1,2], but I expected[1,2,3].')]
}
synthetic=[];expectations={}
for name,cs in controls.items():
    template=next(t for t in original if t['suite']=='presentation_'+name)
    for i,(score,text) in enumerate(cs):
        t=copy.deepcopy(template);pid=f'SYN-{name}-{i}';t.update(pair_id=pid,question_id=pid,task_id=f'presentation_{name}/{pid}/explicit',lines={'Q1':text})
        t['messages'][1]['content']=t['messages'][1]['content'].split('QUESTION LINES:\n')[0]+'QUESTION LINES:\n'+json.dumps(t['lines'])
        for arm in ['baseline','explicit_boundary']:
            u=make(t,arm);synthetic.append(u);expectations[u['task_id']]=score
feature_tasks=natural+synthetic
answers=read(OLD/'answers7/tasks.json');chosen=sorted(answers,key=lambda t:rank(t['task_id']))[:96];formatted=[]
for t in chosen:
    for arm in ['baseline','format_reminder']:
        u=copy.deepcopy(t);u.update(arm=arm,task_id=t['task_id'].rsplit('/',1)[0]+'/'+arm)
        if arm=='format_reminder':
            a,b=u['messages'][1]['content'].split('DATA:\n',1)
            u['messages'][1]['content']=a+'\nOUTPUT CHECK: evidence_ids must contain at most THREE existing answer line IDs. Choose the most informative IDs, not an exhaustive list. Preserve the substantive assessment; this instruction changes only citation formatting. Return valid JSON.\nDATA:\n'+b
        formatted.append(u)
stages=[]
for name,tasks,model,gpu in [('features14',feature_tasks,'14B',0),('features7',feature_tasks,'7B',5),('format7',formatted,'7B',5)]:
    tasks=sorted(tasks,key=lambda t:(rank(t['pair_id']),t['suite'],t['arm']))
    save(R/name/'tasks.json',tasks)
    stages.append(dict(name=name,model=model,gpu=gpu,planned=len(tasks),sha256=hashlib.sha256((R/name/'tasks.json').read_bytes()).hexdigest()))
for name in ['run.py','request_protocol.py','run_helpfulness_grounding.py']:shutil.copyfile(OLD/name,R/name)
save(R/'stages.json',stages);save(R/'expectations.json',expectations)
save(R/'plan.json',dict(natural_question_ids=ids,answer_ids=[t['pair_id'] for t in chosen],stages=stages,planned=sum(s['planned'] for s in stages),reserved_used=False,selection='Fixed SHA256 ranking with explicit-check-314 salt, independent of scores/validity',hypotheses=['Explicit boundaries change inferred-versus-stated feature coding','Known designed absence/detail cases test boundary compliance','Citation reminder improves schema compliance without necessarily improving semantic validity'],limits=['All natural cases reused development data','Synthetic expectations are authored diagnostic criteria, not independent gold labels','Same-family agreement is not accuracy','Explicit-boundary wording changes several instructions, not a single-factor causal study','Invalid outputs remain invalid; no posthoc relaxation','No new regression using unvalidated features'],analysis='Report all statuses; paired natural score transitions; designed-case agreement; citation count vs nonexistent ID vs JSON errors; paired valid helpfulness transitions; baseline comparison to archived runs. Each stage3h hard/2.75h soft maximum; never pad runtime.'))
assert sum(s['planned'] for s in stages)==1416
print(json.dumps(stages))
